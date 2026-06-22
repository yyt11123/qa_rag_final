"""
P1: 样本分流 — Question Unit → Sample Type
===========================================
给每个 question unit 打上 sample_type(五分类),
这是评测集的第一闸门。

输入:
  - intermediate/question_units.json    (P0 产出)
  - real_conversation_recording/sessions/*.json (原始session,用于检测assistant错误)

输出:
  - intermediate/sample_classification.json  — 全部分类结果
  - intermediate/p1_summary.json             — 汇总统计

分流规则(按判定优先级):

  P1   system_error:   assistant 消息命中明确错误特征
       (不看 user 消息——用户问"error是什么意思"不是系统错误)

  P2   chitchat:       P0 标记 is_substantive=false
       (纯寒暄/问候/测试/无实质内容)

  P3   suspected_refusal: 三条同时满足:
       (1) 无文档引用  (references + relevant_docs 均为空)
       (2) 无具体事实  (回答不含数值/条款术语)
       (3) 非 greeting  (排除问候模板)
       ★ 注意:不用关键词匹配"抱歉/无法/找不到"
          这些词出现在正常回答里是高频事件

  P4   multi_doc:      跨保司综合题
       (question涉及多保司比较 / question_types含综合)

  P5   normal_answer:  余下全部

反转说明:
  suspected_refusal 在 P4(检索核实)后可能翻转为:
    - normal_answer:   语料库中有答案 → 系统拒答是错的(如TERM-0007)
    - correct_refusal: 语料库中确实无答案 → 系统拒答是对的

用法:
  python p1_classify.py
  python p1_classify.py --unit-id 144263597516689518-1  # 调试单条
"""

import json
import os
import re
import sys
import glob
import argparse
from collections import Counter, defaultdict
from datetime import datetime


# ============================================================================
# 配置
# ============================================================================

SESSIONS_DIR = "real_conversation_recording/sessions"
P0_UNITS_PATH = "intermediate/question_units.json"
OUTPUT_DIR = "intermediate"


# ============================================================================
# 错误特征检测(system_error)
# ============================================================================

# 只看 assistant 消息,匹配明确错误特征
SYSTEM_ERROR_PATTERNS: list[tuple[str, str]] = [
    # (pattern, description)
    (r"Traceback\s*\(most recent call last\):", "Python traceback"),
    (r"AttributeError:\s*\S", "Python AttributeError"),
    (r"TypeError:\s*\S", "Python TypeError"),
    (r"KeyError:\s*\S", "Python KeyError"),
    (r"ValueError:\s*\S", "Python ValueError"),
    (r"ConnectionError:\s*\S", "Python ConnectionError"),
    (r"RuntimeError:\s*\S", "Python RuntimeError"),
    (r"ImportError:\s*\S", "Python ImportError"),
    (r"IndexError:\s*\S", "Python IndexError"),
    (r"智能体出错了", "固定错误提示-智能体"),
    (r"很抱歉.{0,10}我遇到了一些技术问题", "固定错误提示-技术问题"),
    (r"系统繁忙.{0,10}请稍后重试", "固定错误提示-系统繁忙"),
    (r"服务(暂时|不可|异常|超时|中断)", "服务异常关键词"),
]

# greeting 模板特征(用于区分 chitchat 和 suspected_refusal)
GREETING_TEMPLATE_FEATURES: list[str] = [
    "我是小飞飞",
    "你的保险小助手",
    "专门帮你解答保险",
    "很高兴见到你",
    "今天有什么想聊的",
    "今天有什么想了解",
    "随时告诉我",
    "随时问我",
    "很高兴再次见到你",
    "小飞飞来啦",
    "小飞飞又来啦",
]


# ============================================================================
# 核心判定函数
# ============================================================================

def check_system_error(assistant_contents: list[str]) -> tuple[bool, str | None]:
    """检查 assistant 消息是否包含系统错误特征。
    只在 role=assistant 的消息中查找,不看 user 消息。
    """
    for content in assistant_contents:
        if not content:
            continue
        for pattern, desc in SYSTEM_ERROR_PATTERNS:
            if re.search(pattern, content):
                return True, desc
    return False, None


def check_is_greeting_template(content: str) -> bool:
    """检查是否为系统 greeting 模板(非真正回答)。"""
    for feature in GREETING_TEMPLATE_FEATURES:
        if feature in content:
            return True
    return False


def check_has_factual_content(content: str) -> bool:
    """检查回答是否包含具体事实(数值/条款术语/具体信息)。

    正信号:
      - 金额/比例: \d+[万亿千百]?\s*(%|元|港|美|月|年|天|日|岁)
      - 条款术语: 等待期|保费|保额|现金价值|保证|赔付|保单年度|缴付
      - 具体信息: reference chunk_id 标签
    """
    # 数值+单位
    has_numbers = bool(re.search(
        r'\d+[万亿千百]?\s*(%|％|元|港币?|美[元金]|月|年|天|日|岁|保单|计划)',
        content
    ))
    # 条款/保险术语
    term_terms = [
        "等待期", "保费", "保额", "现金价值", "保证利息", "保证年利率",
        "赔付", "保单年度", "缴付", "预缴", "趸交", "受保人", "投保人",
        "受益人", "持有人", "豁免", "身故", "退保", "保障范围", "不保事项",
        "医保", "重疾", "危疾", "储蓄", "年金", "分红", "回报率",
        "保费折扣", "保费回赠", "保费优惠",
    ]
    has_term_terms = any(term in content for term in term_terms)

    # 有 reference chunk
    has_reference = "<reference" in content

    return has_numbers or has_term_terms or has_reference


def classify_unit(
    unit: dict,
    session_data: dict | None,
) -> dict:
    """
    对单个 question unit 进行分流判定。
    返回带有 sample_type / is_evaluable / reasoning 的分类结果。
    """
    unit_id = unit["unit_id"]
    session_id = unit["session_id"]

    # 获取 assistant 响应内容
    assistant_contents: list[str] = []
    has_references = False
    has_relevant_docs = False

    if session_data:
        msgs = session_data.get("messages", [])
        for msg in msgs:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                assistant_contents.append(content)

                # 检查 references 和 relevant_docs
                refs = msg.get("references", [])
                if refs:
                    has_references = True
                state = msg.get("state", {})
                relevant = state.get("relevant_docs", [])
                if relevant:
                    has_relevant_docs = True

    # ── P1: system_error ──
    is_error, error_desc = check_system_error(assistant_contents)
    if is_error:
        return {
            **unit,
            "sample_type": "system_error",
            "is_evaluable": False,
            "classification_reason": f"assistant消息含错误特征: {error_desc}",
            "classification_confidence": "high",
        }

    # ── P2: chitchat ──
    if not unit.get("is_substantive", True):
        return {
            **unit,
            "sample_type": "chitchat",
            "is_evaluable": False,
            "classification_reason": "P0标记is_substantive=false(纯寒暄/问候/测试)",
            "classification_confidence": "high",
        }

    # ── 辅助: session metadata ──
    session_companies = unit.get("_session_company_names", [])
    session_question_types = []
    if session_data:
        s = session_data.get("session", {})
        session_question_types = s.get("question_types", [])

    # ── P3: multi_doc ──
    # 只看问题内容本身,不继承 session 级的 question_types
    # (session 标"多业务对比"不代表每个 unit 都跨司)
    question = unit.get("question_raw", "")
    question_companies = _detect_companies_in_text(question)

    # 信号1: 问题中出现了 2+ 家保司
    multi_company = len(question_companies) >= 2

    # 信号2: 问题明确要求跨司比较/排名,即使没点名具体公司
    cross_patterns = [
        r"哪[家些个].{0,5}(保司|保险公司|公司|产品).{0,10}(比较|对比|好|优|便宜|划算|强|适合|支持|有)",
        r"对比一下",
        r"比较.{0,3}(一下|哪个|哪家|各家)",
        r"有什么(区别|不同|差异|优劣)",
        r"(哪个|哪家|哪些).{0,5}(更|最).{0,3}(好|优|便宜|划算)",
        r"(综合|汇总|整理).{0,3}(对比|比较|一览|总结)",
        r".{0,5}(保司|保险公司).{0,5}(对比|比较|排名|一览)",
    ]
    cross_question = any(re.search(p, question) for p in cross_patterns)

    is_cross = multi_company or cross_question
    if is_cross:
        return {
            **unit,
            "sample_type": "multi_doc",
            "is_evaluable": True,
            "classification_reason": (
                f"跨保司/综合题: "
                f"question_companies={question_companies}, "
                f"session_companies={session_companies}, "
                f"question_types={session_question_types}"
            ),
            "classification_confidence": "high",
        }

    # ── P4: suspected_refusal ──
    # 三条同时满足才进待核验池
    # (1) 无文档引用
    no_docs = not has_references and not has_relevant_docs
    # (2) 无具体事实
    all_assistant_text = "\n".join(assistant_contents)
    has_facts = check_has_factual_content(all_assistant_text)
    no_facts = not has_facts
    # (3) 非 greeting 模板
    is_greeting = check_is_greeting_template(all_assistant_text)

    if no_docs and no_facts and not is_greeting:
        return {
            **unit,
            "sample_type": "correct_refusal",
            "is_evaluable": True,
            "classification_reason": (
                "疑似拒答(待P4核验): "
                f"无文档引用(refs={has_references},rel_docs={has_relevant_docs}), "
                f"无具体事实, 非greeting模板"
            ),
            "classification_confidence": "medium",  # 待P4确认是否真拒答
            "_refusal_signals": {
                "no_docs": no_docs,
                "no_facts": no_facts,
                "is_greeting": is_greeting,
            },
        }

    # ── P5: normal_answer ──
    # 有 assistant 回复? → normal_answer / 无 → 标记
    if not assistant_contents or all(not c.strip() for c in assistant_contents):
        return {
            **unit,
            "sample_type": "correct_refusal",
            "is_evaluable": True,
            "classification_reason": "无assistant回复内容(可能系统未响应)",
            "classification_confidence": "medium",
        }

    return {
        **unit,
        "sample_type": "normal_answer",
        "is_evaluable": True,
        "classification_reason": "assistant有具体回答",
        "classification_confidence": "high",
    }


def _detect_companies_in_text(text: str) -> list[str]:
    """从文本中检测保司名。按长名优先匹配,避免'太平'和'中国太平'重复计数。"""
    # 按长度降序排列(长名优先),每个位置只匹配一次
    company_keywords = [
        "中国人寿", "中国太平", "周大福人寿", "周大福",
        "太平洋人寿", "太平洋",
        "大都会人寿", "大都会",
        "苏黎世人寿", "苏黎世保险", "苏黎世",
        "中银人寿", "中银",
        "汇丰人寿", "汇丰",
        "恒生保险", "恒生",
        "安达人寿", "安达",
        "永明金融", "永明",
        "富卫保险", "富卫人寿", "富卫",
        "万通保险", "万通",
        "安盛保险", "安盛",
        "保诚保险", "保诚",
        "友邦保险", "友邦",
        "宏利保险", "宏利",
        "立桥人寿", "立桥",
        "富邦保险", "富邦",
        "忠意保险", "忠意",
        "东亚保险", "东亚",
        "国寿",
        "AIA", "AXA", "FWD", "Manulife",
    ]
    found = []
    matched_positions: set[int] = set()
    text_lower = text.lower()

    for kw in company_keywords:
        kw_lower = kw.lower()
        start = 0
        while start < len(text_lower):
            pos = text_lower.find(kw_lower, start)
            if pos == -1:
                break
            # 检查这个位置是否已被更长的匹配覆盖
            positions = set(range(pos, pos + len(kw)))
            if not positions & matched_positions:
                found.append(kw)
                matched_positions.update(positions)
            start = pos + 1

    return found


# ============================================================================
# 批量处理
# ============================================================================

def load_session_index(sessions_dir: str) -> dict[str, dict]:
    """加载所有 session JSON,按 session_id 建索引。"""
    index = {}
    pattern = os.path.join(sessions_dir, "*.json")
    files = sorted(glob.glob(pattern))
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                session = json.load(f)
            s = session.get("session", {})
            sid = s.get("id", "")
            if sid:
                index[sid] = session
        except Exception:
            continue
    return index


def process_all(units: list[dict], session_index: dict[str, dict]) -> dict:
    """批量分流全部 unit。"""
    results: list[dict] = []
    stats = {
        "total": len(units),
        "sample_type_counts": Counter(),
        "confidence_counts": Counter(),
        "suspected_refusal_count": 0,
    }

    for unit in units:
        session_data = session_index.get(unit["session_id"])
        result = classify_unit(unit, session_data)
        results.append(result)

        stats["sample_type_counts"][result["sample_type"]] += 1
        stats["confidence_counts"][result.get("classification_confidence", "unknown")] += 1

        if result["sample_type"] == "correct_refusal":
            stats["suspected_refusal_count"] += 1

    return {"results": results, "stats": stats}


def print_summary(stats: dict) -> None:
    """打印分类汇总。"""
    print("=" * 62)
    print("  P1 样本分流 — 汇总统计")
    print("=" * 62)
    print(f"  总 unit 数:              {stats['total']:>6}")

    st = stats["sample_type_counts"]
    for sample_type in ["normal_answer", "correct_refusal", "multi_doc",
                          "chitchat", "system_error"]:
        count = st.get(sample_type, 0)
        pct = count / stats["total"] * 100 if stats["total"] else 0
        print(f"    {sample_type:<25} {count:>5}  ({pct:>5.1f}%)")

    print(f"  ─────────────────────────────")
    print(f"  待核验数(suspected_refusal): {stats['suspected_refusal_count']:>5}")
    print()
    print("  置信度分布:")
    for conf, count in stats["confidence_counts"].most_common():
        print(f"    {conf:<10} {count:>5}")
    print("=" * 62)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="P1: Sample Classification")
    parser.add_argument("--unit-id", type=str, help="调试单条 unit")
    parser.add_argument("--sessions-dir", type=str, default=SESSIONS_DIR)
    parser.add_argument("--units-path", type=str, default=P0_UNITS_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 P0 输出
    print(f"加载 question units: {args.units_path}")
    with open(args.units_path, "r", encoding="utf-8") as f:
        all_units = json.load(f)
    print(f"  共 {len(all_units)} 条")

    # 加载 session 索引
    print(f"加载 session 索引: {args.sessions_dir}")
    session_index = load_session_index(args.sessions_dir)
    print(f"  共 {len(session_index)} 个 session")

    if args.unit_id:
        # 单条调试
        target = None
        for u in all_units:
            if u["unit_id"] == args.unit_id:
                target = u
                break
        if target is None:
            print(f"未找到 unit: {args.unit_id}")
            sys.exit(1)

        session_data = session_index.get(target["session_id"])
        result = classify_unit(target, session_data)
        print(f"\n[{result['unit_id']}]")
        print(f"  sample_type:    {result['sample_type']}")
        print(f"  is_evaluable:   {result['is_evaluable']}")
        print(f"  confidence:      {result.get('classification_confidence')}")
        print(f"  reason:         {result.get('classification_reason')}")
        print(f"  question_raw:   {result['question_raw'][:150]}")

        if result["sample_type"] == "correct_refusal":
            signals = result.get("_refusal_signals", {})
            print(f"  refusal_signals: {signals}")
        return

    # 全量处理
    print("开始分流...")
    output = process_all(all_units, session_index)

    # 打印汇总
    print_summary(output["stats"])

    # 写入分类结果
    out_path = os.path.join(args.output_dir, "sample_classification.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output["results"], f, ensure_ascii=False, indent=2)
    print(f"\n写入分类结果: {out_path} ({len(output['results'])} 条)")

    # 写入统计
    summary_path = os.path.join(args.output_dir, "p1_summary.json")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "stats": {
            "total": output["stats"]["total"],
            "sample_type_counts": dict(output["stats"]["sample_type_counts"]),
            "confidence_counts": dict(output["stats"]["confidence_counts"]),
            "suspected_refusal_count": output["stats"]["suspected_refusal_count"],
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"写入统计:      {summary_path}")

    # 单独输出 suspected_refusal 列表(便于 P4 优先处理)
    refusals = [r for r in output["results"]
                if r["sample_type"] == "correct_refusal"
                and r.get("classification_confidence") == "medium"]
    if refusals:
        refusal_path = os.path.join(args.output_dir, "p1_medium_confidence_refusals.json")
        with open(refusal_path, "w", encoding="utf-8") as f:
            json.dump(refusals, f, ensure_ascii=False, indent=2)
        print(f"写入疑似拒答:  {refusal_path} ({len(refusals)} 条)")

    print("\nDone.")


if __name__ == "__main__":
    main()
