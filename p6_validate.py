"""
P6: Schema 校验 — Validate Against schema.json
================================================
逐条校验样本是否符合 schema.json 的约束条件。

输入:
  - intermediate/answer_annotated.json  (P5 产出,完整标注集)
  - schema.json                         (项目根目录)

输出:
  - intermediate/validation_report.json  — 详细校验报告
  - intermediate/p6_summary.json         — 校验统计

校验规则:
  1. required 9个必填字段
  2. additionalProperties: false(无额外字段)
  3. enum 约束(sample_type / expected_behavior / eval_method)
  4. hash pattern: ^[a-f0-9]{40}$
  5. page >= 1
  6. allOf 条件约束:
     a. refuse → base_answer 为空
     b. clarify → 有 clarify_question
     c. objective_qa → 有 base_answer + required_keywords
     d. multi_point → 有 key_points
     e. is_evaluable=true + normal_answer/multi_doc → 有 gold_evidence

用法:
  python p6_validate.py
  python p6_validate.py --strict  # 严格模式(additionalProperties检查)
"""

import json
import os
import re
import argparse
from collections import Counter
from datetime import datetime


# ============================================================================
# 配置
# ============================================================================

ANNOTATED_PATH = "intermediate/answer_annotated.json"
SCHEMA_PATH = "schema.json"
OUTPUT_DIR = "intermediate"

# schema 定义的必填字段
REQUIRED_FIELDS: list[str] = [
    "id", "session_id", "sample_type", "is_evaluable",
    "kb_version", "question_raw", "expected_behavior",
    "eval_method", "verified",
]

# schema 定义的合法枚举值
VALID_SAMPLE_TYPES: set[str] = {
    "normal_answer", "correct_refusal", "multi_doc", "chitchat", "system_error",
}
VALID_EXPECTED_BEHAVIORS: set[str] = {
    "answer", "refuse", "clarify", "correct_premise",
}
VALID_EVAL_METHODS: set[str] = {
    "objective_qa", "refusal", "clarify", "multi_point",
}

# schema properties — 所有合法字段
ALLOWED_FIELDS: set[str] = {
    # A 层
    "id", "session_id", "sample_type", "is_evaluable",
    "source_company", "source_company_id", "product_name", "product_id",
    "kb_version", "org_code",
    # B 层
    "question_raw", "question_lang", "question_type", "question_sub_type",
    "question_quality",
    # C 层
    "gold_evidence", "requires_multi_doc", "no_answer_in_corpus",
    "retrieval_granularity", "retrieval_candidates_snapshot",
    "kb_version_for_evidence",
    # D 层
    "expected_behavior", "base_answer", "required_keywords",
    "acceptable_answers", "key_points", "forbidden_content",
    "currency", "currency_required", "clarify_question",
    "expected_fallback",  # P5 新增,非 schema 标准字段
    # E 层
    "eval_method", "score_weights",
    # F 层
    "annotator", "verified", "notes",
}


# ============================================================================
# 校验函数
# ============================================================================

def validate_sample(sample: dict, strict: bool = False) -> list[dict]:
    """校验单条样本,返回错误列表。"""
    errors: list[dict] = []

    # 1. 必填字段
    for field in REQUIRED_FIELDS:
        if field not in sample or sample[field] is None:
            errors.append({
                "type": "missing_required",
                "field": field,
                "message": f"缺少必填字段 {field}",
            })

    # 2. additionalProperties(严格模式)
    if strict:
        for key in sample:
            if key.startswith("_"):
                continue  # 内部字段跳过
            if key not in ALLOWED_FIELDS:
                errors.append({
                    "type": "unknown_field",
                    "field": key,
                    "message": f"未知字段 {key}(不在 schema 定义中)",
                })

    # 3. enum 约束
    st = sample.get("sample_type")
    if st and st not in VALID_SAMPLE_TYPES:
        errors.append({
            "type": "invalid_enum",
            "field": "sample_type",
            "message": f"非法值: {st},合法值: {VALID_SAMPLE_TYPES}",
        })

    eb = sample.get("expected_behavior")
    if eb and eb not in VALID_EXPECTED_BEHAVIORS:
        errors.append({
            "type": "invalid_enum",
            "field": "expected_behavior",
            "message": f"非法值: {eb},合法值: {VALID_EXPECTED_BEHAVIORS}",
        })

    em = sample.get("eval_method")
    if em and em not in VALID_EVAL_METHODS:
        errors.append({
            "type": "invalid_enum",
            "field": "eval_method",
            "message": f"非法值: {em},合法值: {VALID_EVAL_METHODS}",
        })

    # 4. hash pattern
    gold = sample.get("gold_evidence", [])
    for i, g in enumerate(gold):
        h = g.get("hash", "")
        if h and not re.match(r'^[a-f0-9]{40}$', h):
            errors.append({
                "type": "invalid_hash",
                "field": f"gold_evidence[{i}].hash",
                "message": f"hash格式不对(应为40位hex): {h}",
            })
        if g.get("page", 0) < 1:
            errors.append({
                "type": "invalid_page",
                "field": f"gold_evidence[{i}].page",
                "message": f"page必须>=1,实际: {g.get('page')}",
            })

    # 5. allOf 条件约束
    _validate_conditional(sample, errors)

    return errors


def _validate_conditional(sample: dict, errors: list[dict]) -> None:
    """校验 schema.json 的 allOf 条件约束。"""
    behavior = sample.get("expected_behavior", "")
    eval_method = sample.get("eval_method", "")
    is_eval = sample.get("is_evaluable", False)
    sample_type = sample.get("sample_type", "")

    # 5a. refuse → base_answer 为空
    if behavior == "refuse":
        ba = sample.get("base_answer", "")
        if ba:
            errors.append({
                "type": "conditional_violation",
                "condition": "refuse → base_answer应为空",
                "message": f"拒答题的 base_answer 应为空,实际: '{ba[:50]}...'",
            })

    # 5b. clarify → 有 clarify_question
    if behavior == "clarify":
        cq = sample.get("clarify_question")
        if not cq:
            errors.append({
                "type": "conditional_violation",
                "condition": "clarify → 需要 clarify_question",
                "message": "clarify 题型缺少 clarify_question",
            })

    # 5c. objective_qa → 有 base_answer + required_keywords
    if eval_method == "objective_qa":
        if not sample.get("base_answer"):
            errors.append({
                "type": "conditional_violation",
                "condition": "objective_qa → 需要 base_answer",
                "message": "objective_qa 题缺少 base_answer",
            })
        if not sample.get("required_keywords"):
            errors.append({
                "type": "conditional_violation",
                "condition": "objective_qa → 需要 required_keywords",
                "message": "objective_qa 题缺少 required_keywords",
            })

    # 5d. multi_point → 有 key_points
    if eval_method == "multi_point":
        if not sample.get("key_points"):
            errors.append({
                "type": "conditional_violation",
                "condition": "multi_point → 需要 key_points",
                "message": "multi_point 题缺少 key_points",
            })

    # 5e. is_evaluable=true + normal_answer/multi_doc → 有 gold_evidence
    if is_eval and sample_type in ("normal_answer", "multi_doc"):
        if not sample.get("gold_evidence"):
            errors.append({
                "type": "conditional_violation",
                "condition": (
                    "is_evaluable=true + normal_answer/multi_doc → 需要 gold_evidence"
                ),
                "message": "可评测样本缺少 gold_evidence",
            })


# ============================================================================
# 批量校验
# ============================================================================

def validate_all(samples: list[dict], strict: bool = False) -> dict:
    """批量校验全部样本。"""
    report: list[dict] = []
    error_counts = Counter()
    total_errors = 0
    clean_count = 0
    error_count = 0

    for s in samples:
        sample_errors = validate_sample(s, strict=strict)
        internal_fields = [k for k in s if k.startswith("_")]

        entry = {
            "id": s.get("id", "unknown"),
            "session_id": s.get("session_id", ""),
            "sample_type": s.get("sample_type", ""),
            "has_errors": len(sample_errors) > 0,
            "error_count": len(sample_errors),
            "errors": sample_errors,
            "internal_fields_remaining": internal_fields,
        }
        report.append(entry)

        total_errors += len(sample_errors)
        if sample_errors:
            error_count += 1
        else:
            clean_count += 1

        for e in sample_errors:
            error_counts[e["type"]] += 1

    return {
        "report": report,
        "summary": {
            "total_samples": len(samples),
            "clean": clean_count,
            "with_errors": error_count,
            "total_errors": total_errors,
            "error_type_distribution": dict(error_counts),
        },
    }


def print_summary(result: dict) -> None:
    """打印校验总结。"""
    s = result["summary"]

    print("=" * 62)
    print("  P6 Schema 校验 — 报告")
    print("=" * 62)
    print(f"  总样本:       {s['total_samples']}")
    print(f"  通过:         {s['clean']}")
    print(f"  有错误:       {s['with_errors']}")
    print(f"  总错误数:     {s['total_errors']}")
    print()

    if s["total_errors"] > 0:
        print("  错误类型分布:")
        for etype, count in s["error_type_distribution"].items():
            print(f"    {etype:<30} {count:>4}")
    else:
        print("  ✓ 全部样本通过 Schema 校验")

    print("=" * 62)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="P6: Schema Validation")
    parser.add_argument("--annotated", type=str, default=ANNOTATED_PATH)
    parser.add_argument("--schema", type=str, default=SCHEMA_PATH)
    parser.add_argument("--strict", action="store_true",
                        help="严格模式(检查 additionalProperties)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载样本
    print(f"加载样本: {args.annotated}")
    try:
        with open(args.annotated, "r", encoding="utf-8") as f:
            samples = json.load(f)
    except FileNotFoundError:
        # 回退: P4 → P3 → P2
        for fallback in ["retrieval_verified.json", "question_annotated.json",
                          "sampled_candidates.json"]:
            fb_path = os.path.join(args.output_dir, fallback)
            if os.path.exists(fb_path):
                with open(fb_path, "r", encoding="utf-8") as f:
                    samples = json.load(f)
                print(f"  回退到: {fb_path}")
                break
        else:
            print("错误: 找不到任何可校验的样本文件")
            sys.exit(1)

    print(f"  共 {len(samples)} 条")

    # 校验
    result = validate_all(samples, strict=args.strict)
    print_summary(result)

    # 写入报告
    report_path = os.path.join(args.output_dir, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result["report"], f, ensure_ascii=False, indent=2)
    print(f"\n写入详细报告: {report_path}")

    # 写入统计
    summary_path = os.path.join(args.output_dir, "p6_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "strict_mode": args.strict,
            **result["summary"],
        }, f, ensure_ascii=False, indent=2)
    print(f"写入统计:     {summary_path}")

    # 如果全部通过
    if result["summary"]["total_errors"] == 0:
        print("\n✓ 全部校验通过,可以进入 P7 双人核验")

    print("\nDone.")


if __name__ == "__main__":
    main()
