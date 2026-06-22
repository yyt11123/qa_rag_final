"""
P5: 答案标注框架 — Answer Ground Truth
========================================
为每条样本标注生成层的 ground truth。

输入:
  - intermediate/retrieval_verified.json  (P4 产出,含 gold_evidence)

输出:
  - intermediate/answer_annotated.json     — 完整标注集(P5完成)
  - intermediate/p5_annotation_guide.md    — 标注指引
  - intermediate/p5_summary.json           — 标注统计

标注策略(按 sample_type/expected_behavior):

  题型                  eval_method     核心标注字段
  ──────────────────────────────────────────────────────
  normal_answer         objective_qa    base_answer + required_keywords
                                        + acceptable_answers + forbidden_content
  correct_refusal       refusal         forbidden_content + expected_fallback(★新增)
  multi_doc             multi_point     key_points
  clarify              clarify         clarify_question

forbidden_content 来源:
  P4 产出的 _p5_forbidden_hints(irrelevant_high_score 列表)
  → 标注员从中提炼不应出现的幻觉内容

用法:
  python p5_annotate_answers.py
"""

import json
import os
import argparse
from collections import Counter
from datetime import datetime


# ============================================================================
# 配置
# ============================================================================

VERIFIED_PATH = "intermediate/retrieval_verified.json"
OUTPUT_DIR = "intermediate"


# ============================================================================
# 主流程
# ============================================================================

def annotate_answers(candidates: list[dict]) -> list[dict]:
    """为每条候选样本准备答案标注字段(留空待人工填)。"""
    annotated = []

    for c in candidates:
        st = c.get("sample_type", "")
        behavior = c.get("expected_behavior", "answer")

        # 根据题型设置标注模板
        if st == "correct_refusal":
            # 此时应该已被 P4 翻转,未翻转的保留标记
            c["expected_behavior"] = "refuse"
            c["eval_method"] = "refusal"
            c["base_answer"] = ""
            c["required_keywords"] = []
            c["acceptable_answers"] = []
            c["expected_fallback"] = ""   # ★新增: 预期兜底引导

        elif st == "normal_answer" and behavior == "answer":
            c["expected_behavior"] = "answer"
            c["eval_method"] = "objective_qa"
            c["score_weights"] = {"base": 0.25, "keyword": 0.25, "similarity": 0.5}

        elif st == "multi_doc":
            c["expected_behavior"] = "answer"
            c["eval_method"] = "multi_point"
            c["key_points"] = c.get("key_points", [])

        elif st == "correct_refusal":
            c["expected_behavior"] = "refuse"
            c["eval_method"] = "refusal"
            c["base_answer"] = ""
            c["required_keywords"] = []
            c["acceptable_answers"] = []

        # 标注指引(供标注员参考)
        hints = c.pop("_p5_forbidden_hints", [])
        c["_p5_annotation_guide"] = {
            "_status": "pending",  # pending → in_progress → done → reviewed
            "_annotator": "",
            "_forbidden_hints_from_p4": [  # P4发现的高分不相关chunk
                {
                    "file_name": h.get("file_name", ""),
                    "page": h.get("page", ""),
                    "score": h.get("score", 0),
                    "wrong_info": h.get("wrong_info", h.get("text_preview", "")),
                    "why_irrelevant": h.get("why_irrelevant", ""),
                }
                for h in hints
            ],
            "_annotation_checklist": _build_checklist(c),
        }

        annotated.append(c)

    return annotated


def _build_checklist(candidate: dict) -> list[str]:
    """根据题型生成标注核查清单。"""
    st = candidate.get("sample_type", "")
    checklist = []

    if st == "normal_answer":
        checklist = [
            "□ base_answer: 核心事实是否正确?是否限定了版本/年期/币种?",
            "□ required_keywords: 3-5个关键实体,错则一票否决?",
            "□ acceptable_answers: 2-3个等价表述?",
            "□ forbidden_content: 对照P4的_forbidden_hints_from_p4填写禁止出现的幻觉内容",
            "□ currency: 涉及金额/保额时是否标了币种(USD/HKD)?",
            "□ notes: 标注出题意图和边界判断",
        ]
    elif st == "correct_refusal":
        checklist = [
            "□ forbidden_content: 对照P4的_forbidden_hints_from_p4填写禁编造内容",
            "□ expected_fallback: 理想情况应该给什么兜底引导?",
            "    (例:'建议查看产品条款第X章/联系XX客服/参考XX官网')",
            "□ notes: 注明为什么语料库没有答案",
        ]
    elif st == "multi_doc":
        checklist = [
            "□ key_points: 逐条列出信息点,每个点独立计分",
            "□ 每个key_point有对应的gold_evidence出处",
            "□ notes: 注明跨文档验证情况",
        ]
    elif st == "clarify":
        checklist = [
            "□ clarify_question: 理想的反问是否精准捕捉了歧义点?",
            "□ notes: 注明为什么需要澄清",
        ]

    checklist.append("□ verified: 是否已双人核验通过?")
    return checklist


def generate_stats(annotated: list[dict]) -> dict:
    """生成标注统计。"""
    stats = {
        "total": len(annotated),
        "eval_method_distribution": Counter(),
        "expected_behavior_distribution": Counter(),
        "sample_type_distribution": Counter(),
        "has_base_answer": 0,
        "has_keywords": 0,
        "has_acceptable_answers": 0,
        "has_forbidden_content": 0,
        "has_key_points": 0,
        "has_expected_fallback": 0,
        "has_clarify_question": 0,
        "currency_required": 0,
    }

    for c in annotated:
        stats["eval_method_distribution"][c.get("eval_method", "unknown")] += 1
        stats["expected_behavior_distribution"][c.get("expected_behavior", "unknown")] += 1
        stats["sample_type_distribution"][c.get("sample_type", "unknown")] += 1

        if c.get("base_answer"):
            stats["has_base_answer"] += 1
        if c.get("required_keywords"):
            stats["has_keywords"] += 1
        if c.get("acceptable_answers"):
            stats["has_acceptable_answers"] += 1
        if c.get("forbidden_content"):
            stats["has_forbidden_content"] += 1
        if c.get("key_points"):
            stats["has_key_points"] += 1
        if c.get("expected_fallback"):
            stats["has_expected_fallback"] += 1
        if c.get("clarify_question"):
            stats["has_clarify_question"] += 1
        if c.get("currency_required"):
            stats["currency_required"] += 1

    return stats


def generate_guide(stats: dict) -> str:
    """生成 P5 标注指引。"""
    return f"""# P5 答案标注指引

> 生成时间: {datetime.now().isoformat()}
> 待标注样本数: {stats['total']}

---

## 标注总览

| 指标 | 值 |
|---|---|
| 总样本数 | {stats['total']} |
| objective_qa 题 | {stats['eval_method_distribution'].get('objective_qa', 0)} |
| refusal 题 | {stats['eval_method_distribution'].get('refusal', 0)} |
| multi_point 题 | {stats['eval_method_distribution'].get('multi_point', 0)} |
| clarify 题 | {stats['eval_method_distribution'].get('clarify', 0)} |

## 逐题型标注要求

### objective_qa (正常有答案)

```
base_answer: 核心事实一句话。限定版本/年期/币种。错则一票否决判0分。

  好: "宏挚传承2年缴付期,第2个保单年度保证利息10%,保单货币为美元"
  差: "保证利息10%"

required_keywords: 3-5个必须出现的关键实体,命中加分。
  例: ["宏挚传承","2年缴","保证利息"]

acceptable_answers: 2-3个等价表述,算语义相似度取上限。

forbidden_content: 对照 _p5_annotation_guide._forbidden_hints_from_p4,
  把高分但不相关chunk里的错误数值/结论提炼进来。
  例: ["6%","6.5%"] ← 这是3年缴的数字,2年缴答这个即错

currency / currency_required: 涉及金额/保额时必标
```

### refusal (正确拒答) ★含 expected_fallback

```
forbidden_content: 禁止编造的内容(特别是高分不相关chunk里的错误结论)

expected_fallback: ★新增字段
  理想的兜底引导。好的拒答: "我没有这个信息,建议您联系XX/查看YY"
  评分时: 纯拒答无引导 → 低分 / 拒答+给出路 → 高分
  例: "建议查阅产品条款第X章,或联系宏利客服热线"
```

### multi_point (综合多文档)

```
key_points: 逐条列出信息点,每个点独立计分
  例: ["要点1: 宏利支持...","要点2: 友邦不支持..."]

每个 key_point 应有对应的 gold_evidence 出处
```

### clarify (需澄清)

```
clarify_question: 理想的澄清反问
  例: "请问您指的是哪个缴付年期?2年缴还是5年缴?"
```

## forbidden_content 填写技巧

标注员请先查看 `_p5_annotation_guide._forbidden_hints_from_p4`——
这些是 P4 检索时发现的高分但不相关的chunk。
模型可能从中提取错误数值/结论用于生成答案。
把这些错误信息填入 forbidden_content,比凭空想准得多。
"""


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="P5: Answer Annotation")
    parser.add_argument("--verified", type=str, default=VERIFIED_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载
    print(f"加载: {args.verified}")
    try:
        with open(args.verified, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    except FileNotFoundError:
        # P4 还没跑,用 P3 的代替
        fallback = os.path.join(args.output_dir, "question_annotated.json")
        print(f"  文件不存在,回退到: {fallback}")
        with open(fallback, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    print(f"  共 {len(candidates)} 条")

    # 标注准备
    annotated = annotate_answers(candidates)
    stats = generate_stats(annotated)

    print(f"  标注模板已生成:")
    print(f"    objective_qa: {stats['eval_method_distribution'].get('objective_qa', 0)}")
    print(f"    refusal:      {stats['eval_method_distribution'].get('refusal', 0)}")
    print(f"    multi_point:  {stats['eval_method_distribution'].get('multi_point', 0)}")
    print(f"    clarify:      {stats['eval_method_distribution'].get('clarify', 0)}")

    # 写入
    out_path = os.path.join(args.output_dir, "answer_annotated.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, ensure_ascii=False, indent=2)
    print(f"\n写入: {out_path}")

    guide_path = os.path.join(args.output_dir, "p5_annotation_guide.md")
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(generate_guide(stats))
    print(f"写入指引: {guide_path}")

    summary_path = os.path.join(args.output_dir, "p5_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "stats": {
                k: dict(v) if isinstance(v, Counter) else v
                for k, v in stats.items()
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"写入统计: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
