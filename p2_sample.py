"""
P2: 分层抽样 — Classified Units → Sampled Candidates
======================================================
从 P1 分流后的各类样本中按比例分层抽取,构成评测集候选。

输入:
  - intermediate/sample_classification.json  (P1 产出,含 sample_type)
  - intermediate/question_units.json         (P0 产出,含 question_raw 等)

输出:
  - intermediate/sampled_candidates.json      — 抽样候选集(schema 雏形)
  - intermediate/p2_summary.json              — 抽样统计

抽样策略:

  sample_type       池大小  抽取数  抽样维度
  ─────────────────────────────────────────────
  normal_answer      1879   80-100  保司×question_type×时间 均匀分层
  suspected_refusal    81   30-40   全部进入候选(P4后再确认)
  multi_doc           239   20-30   优先典型跨司比较题
  chitchat            246    5-10   随机
  system_error         50    5-10   随机

编号规则:
  纯流水号 EVAL-0001 ~ EVAL-NNNN,不含类型前缀。
  样本类型翻转时只需改 sample_type 字段,不改 id。

用法:
  python p2_sample.py
  python p2_sample.py --target 200    # 自定义总目标数
"""

import json
import os
import random
import argparse
from collections import Counter, defaultdict
from datetime import datetime


# ============================================================================
# 配置
# ============================================================================

P1_RESULTS_PATH = "intermediate/sample_classification.json"
P0_UNITS_PATH = "intermediate/question_units.json"
OUTPUT_DIR = "intermediate"

# 总目标样本数
TARGET_TOTAL = 160

# 各类配额(可调)
QUOTA: dict[str, int] = {
    "normal_answer": 85,
    "correct_refusal": 35,   # 全进候选,P4后翻转
    "multi_doc": 25,
    "chitchat": 8,
    "system_error": 7,
}

# 保司最小覆盖(每家至少 N 条 normal_answer)
MIN_PER_COMPANY = 2

# 随机种子(可复现)
RANDOM_SEED = 42


# ============================================================================
# 分层抽样核心
# ============================================================================

def stratified_sample_normal_answer(
    units: list[dict],
    quota: int,
) -> list[dict]:
    """
    normal_answer 分层抽样:
      维度1: 保司(19家至少各2条)
      维度2: question_type(均匀分布)
      维度3: 时间(跨月份均匀)
    """
    random.seed(RANDOM_SEED)

    # 按保司分组
    by_company: dict[str, list[dict]] = defaultdict(list)
    no_company: list[dict] = []

    for u in units:
        company = u.get("context_company")
        if company and isinstance(company, str):
            by_company[company].append(u)
        else:
            no_company.append(u)

    selected: list[dict] = []
    selected_ids: set[str] = set()

    # Step 1: 每家保司保底 MIN_PER_COMPANY 条
    for company, pool in sorted(by_company.items()):
        if len(pool) >= MIN_PER_COMPANY:
            # 在保司内部按 question_type 分层抽
            subsample = _sample_by_question_type(pool, MIN_PER_COMPANY)
        else:
            subsample = list(pool)
        for u in subsample:
            if u["unit_id"] not in selected_ids:
                selected.append(u)
                selected_ids.add(u["unit_id"])

    # Step 2: 从无保司标记的池中补一些
    random.shuffle(no_company)
    for u in no_company:
        if len(selected) >= quota:
            break
        if u["unit_id"] not in selected_ids:
            selected.append(u)
            selected_ids.add(u["unit_id"])

    # Step 3: 如果还不够配额,从已选保司的池中按 question_type 多样补
    if len(selected) < quota:
        remaining = [u for u in units
                     if u["unit_id"] not in selected_ids]
        # 按 question_type 分层补
        extras = _sample_by_question_type(remaining, quota - len(selected))
        for u in extras:
            if len(selected) >= quota:
                break
            if u["unit_id"] not in selected_ids:
                selected.append(u)
                selected_ids.add(u["unit_id"])

    # 截断到配额
    return selected[:quota]


def _sample_by_question_type(units: list[dict], target: int) -> list[dict]:
    """在给定 pool 内按 question_type 均匀抽样。"""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for u in units:
        qt = u.get("question_type", "未知")
        by_type[qt].append(u)

    selected: list[dict] = []
    types = list(by_type.keys())

    # 轮询:每轮从每种 type 各取1条
    while len(selected) < target and types:
        took_any = False
        for qt in list(types):
            if len(selected) >= target:
                break
            if by_type[qt]:
                selected.append(by_type[qt].pop(0))
                took_any = True
            else:
                types.remove(qt)
        if not took_any:
            break

    return selected


def sample_other_types(
    units: list[dict],
    quota: int,
    sample_type: str,
) -> list[dict]:
    """对非 normal_answer 类的简单抽样。"""
    random.seed(RANDOM_SEED)
    pool = list(units)

    if len(pool) <= quota:
        return pool

    # suspected_refusal: 全取(后续 P4 核实)
    if sample_type == "correct_refusal":
        return pool

    # multi_doc: 优先保留有明确跨司比较的
    if sample_type == "multi_doc":
        explicit = [u for u in pool if len(_get_companies_in_question(u)) >= 2]
        implicit = [u for u in pool if len(_get_companies_in_question(u)) < 2]
        random.shuffle(explicit)
        random.shuffle(implicit)
        combined = explicit + implicit
        return combined[:quota]

    # chitchat / system_error: 随机
    random.shuffle(pool)
    return pool[:quota]


def _get_companies_in_question(unit: dict) -> list[str]:
    """从 unit 的 question_raw 中提取保司名。"""
    # 复用 P1 逻辑的简化版
    question = unit.get("question_raw", "")
    # 简单检测
    companies = [
        "友邦", "宏利", "保诚", "安盛", "万通", "富卫",
        "周大福", "中国人寿", "国寿", "中国太平", "太平",
        "中银", "汇丰", "恒生", "安达", "永明",
        "苏黎世", "立桥", "富邦", "大都会", "忠意", "东亚",
        "太平洋", "AIA", "AXA", "FWD", "Manulife",
    ]
    found = []
    for c in companies:
        if c in question:
            found.append(c)
    # 去重(中国太平和太平)
    unique = []
    for c in found:
        # 如果这是短名且已被长名覆盖,跳过
        is_subsumed = any(
            c != other and c in other and other in found
            for other in found
        )
        if not is_subsumed or c not in unique:
            unique.append(c)
    return unique


# ============================================================================
# Schema 雏形填充
# ============================================================================

def build_candidate(unit: dict, eval_id: str, index: int) -> dict:
    """
    将 P0/P1 的内部 unit 转换为 schema.json 格式的样本雏形。
    自动填充可从现有数据推导的字段,检索/答案字段留空待后续阶段。
    """
    # A 层:样本管理
    candidate = {
        "id": eval_id,
        "session_id": unit["session_id"],
        "sample_type": unit["sample_type"],
        "is_evaluable": unit.get("is_evaluable", True),
        "source_company": _resolve_source_company(unit),
        "source_company_id": None,          # P4/P5 通过中台补充(可空)
        "product_name": _resolve_product_name(unit),
        "product_id": None,                  # P4/P5 通过中台补充(可空)
        "kb_version": "rag_v1_260604_bge_m3",
        "org_code": "awm",

        # B 层:输入
        "question_raw": unit.get("question_raw", ""),
        "question_lang": _detect_lang(unit.get("question_raw", "")),
        "question_type": unit.get("question_type", ""),
        "question_sub_type": unit.get("question_sub_type", None),
        "question_quality": {
            "is_underspecified": False,
            "has_false_premise": False,
            "is_multi_intent": False,
            "is_cross_company": unit["sample_type"] == "multi_doc",
            "is_colloquial": False,
            "has_typo": False,
            "needs_term_mapping": False,
        },

        # C 层:检索(留空,待 P4)
        "gold_evidence": [],
        "requires_multi_doc": unit["sample_type"] == "multi_doc",
        "no_answer_in_corpus": False,
        "retrieval_granularity": "hash_page",
        "retrieval_candidates_snapshot": [],
        "kb_version_for_evidence": "rag_v1_260604_bge_m3",

        # D 层:生成(留空,待 P5)
        "expected_behavior": _default_behavior(unit["sample_type"]),
        "base_answer": "",
        "required_keywords": [],
        "acceptable_answers": [],
        "key_points": [],
        "forbidden_content": [],
        "currency": None,
        "currency_required": False,
        "clarify_question": None,

        # E 层:评分
        "eval_method": _default_eval_method(unit["sample_type"]),
        "score_weights": {"base": 0.25, "keyword": 0.25, "similarity": 0.5},

        # F 层:元信息
        "annotator": "",
        "verified": False,
        "notes": "",

        # 内部字段(标注辅助)
        "_p0_unit_id": unit["unit_id"],
        "_p1_reason": unit.get("classification_reason", ""),
        "_sample_index": index,
    }

    return candidate


def _resolve_source_company(unit: dict) -> str | None:
    """解析 source_company: multi_doc 留空,其余取 context_company。"""
    if unit["sample_type"] == "multi_doc":
        return None  # 跨保司题不填单个保司
    company = unit.get("context_company")
    if company and isinstance(company, str):
        return company
    sesh_companies = unit.get("_session_company_names", [])
    if len(sesh_companies) == 1:
        return sesh_companies[0]
    return None


def _resolve_product_name(unit: dict) -> str | None:
    """解析 product_name。"""
    product = unit.get("context_product")
    if product and isinstance(product, str):
        return product
    sesh_products = unit.get("_session_product_names", [])
    if len(sesh_products) == 1:
        return sesh_products[0]
    return None


def _detect_lang(text: str) -> str:
    """简单语言检测。"""
    if not text:
        return "zh-Hans"
    # 繁体字
    trad_chars = set("爲什麼麼爲甚麼爲爲甚爲爲爲爲爲爲爲爲爲爲爲爲爲爲爲")
    trad_signal = any(c in text for c in "爲麼爲後衞護險計劃")
    cantonese_signal = any(w in text for w in ["係", "咁", "嘅", "㗎", "喺", "哋", "嘢", "點解", "邊個"])
    if cantonese_signal:
        return "粤语夹杂"
    if trad_signal:
        return "zh-Hant"
    return "zh-Hans"


def _default_behavior(sample_type: str) -> str:
    if sample_type == "correct_refusal":
        return "refuse"
    return "answer"


def _default_eval_method(sample_type: str) -> str:
    if sample_type == "correct_refusal":
        return "refusal"
    if sample_type == "multi_doc":
        return "multi_point"
    return "objective_qa"


# ============================================================================
# 主流程
# ============================================================================

def sample(p1_results: list[dict], quota: dict[str, int]) -> dict:
    """执行分层抽样并生成 schema 雏形。"""
    random.seed(RANDOM_SEED)

    # 按 sample_type 分组
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in p1_results:
        by_type[r["sample_type"]].append(r)

    # 各类抽样
    sampled: list[dict] = []
    stats: dict[str, dict] = {}

    type_order = ["normal_answer", "correct_refusal", "multi_doc",
                  "chitchat", "system_error"]

    for st in type_order:
        pool = by_type.get(st, [])
        quota_n = quota.get(st, 0)

        if st == "normal_answer":
            picked = stratified_sample_normal_answer(pool, quota_n)
        else:
            picked = sample_other_types(pool, quota_n, st)

        sampled.extend(picked)
        stats[st] = {
            "pool_size": len(pool),
            "quota": quota_n,
            "sampled": len(picked),
        }

    # 转换为 schema 雏形并分配 EVAL-xxxx ID
    candidates = []
    total_digits = max(4, len(str(len(sampled))) + 1)
    for i, unit in enumerate(sampled, 1):
        eval_id = f"EVAL-{i:0{total_digits}d}"
        candidate = build_candidate(unit, eval_id, i)
        candidates.append(candidate)

    return {
        "candidates": candidates,
        "stats": stats,
        "total_sampled": len(candidates),
    }


def print_summary(result: dict) -> None:
    """打印抽样统计。"""
    print("=" * 62)
    print("  P2 分层抽样 — 汇总统计")
    print("=" * 62)

    for st, info in result["stats"].items():
        pool = info["pool_size"]
        quota = info["quota"]
        sampled = info["sampled"]
        pct = sampled / pool * 100 if pool else 0
        print(f"  {st:<25} 池{pool:>4} → 抽{sampled:>3} ({pct:>5.1f}%) 配额{quota}")

    print(f"  ─────────────────────────────")
    print(f"  总计:                {result['total_sampled']} 条候选样本")
    print()

    # 保司覆盖
    candidates = result["candidates"]
    comp_counts = Counter(
        c.get("source_company") for c in candidates
        if c.get("source_company")
    )
    print(f"  保司覆盖: {len(comp_counts)} 家")
    for comp, count in comp_counts.most_common():
        print(f"    {comp:<16} {count:>3}")
    print()

    # sample_type 分布
    st_counts = Counter(c["sample_type"] for c in candidates)
    print("  样本类型分布:")
    for st, c in st_counts.most_common():
        print(f"    {st:<25} {c:>3}")
    print("=" * 62)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="P2: Stratified Sampling")
    parser.add_argument("--target", type=int, default=TARGET_TOTAL,
                        help=f"目标总样本数(默认 {TARGET_TOTAL})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"随机种子(默认 {RANDOM_SEED})")
    parser.add_argument("--p1-results", type=str, default=P1_RESULTS_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 P1 结果
    print(f"加载 P1 分类结果: {args.p1_results}")
    with open(args.p1_results, "r", encoding="utf-8") as f:
        p1_results = json.load(f)
    print(f"  共 {len(p1_results)} 条")

    # 计算实际配额
    quota = dict(QUOTA)
    if args.target != TARGET_TOTAL:
        scale = args.target / sum(quota.values())
        for k in quota:
            quota[k] = max(1, round(quota[k] * scale))
        diff = args.target - sum(quota.values())
        quota["normal_answer"] += diff
        print(f"  调整配额: {quota}")

    # 执行抽样
    result = sample(p1_results, quota)

    # 打印汇总
    print_summary(result)

    # 写入候选集
    out_path = os.path.join(args.output_dir, "sampled_candidates.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result["candidates"], f, ensure_ascii=False, indent=2)
    print(f"\n写入候选集: {out_path} ({len(result['candidates'])} 条)")

    # 写入统计
    summary_path = os.path.join(args.output_dir, "p2_summary.json")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "target_total": args.target,
        "total_sampled": result["total_sampled"],
        "stats": result["stats"],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"写入统计:    {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
