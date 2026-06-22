"""
P3: 问题质量标注框架 — Question Quality Annotation
=====================================================
自动填充可检测的 question_quality 子字段,生成待人工标注的模板。

输入:
  - intermediate/sampled_candidates.json  (P2 产出,含 schema 雏形)

输出:
  - intermediate/question_annotated.json   — 标注后的候选集
  - intermediate/p3_annotation_guide.md    — 标注指引(给标注员看)
  - intermediate/p3_summary.json           — 标注统计

自动化部分:
  - question_lang:         简体/繁体/粤语夹杂
  - question_type:         从 session 继承
  - is_cross_company:      从 sample_type 推断
  - is_colloquial:         粤语特征词检测
  - has_typo:              疑似错别字标记(不确定的标 low confidence)

人工部分(标注员逐条判定):
  - is_underspecified:     信息不全(没说哪个计划/年期)
  - has_false_premise:     预设错误前提
  - is_multi_intent:       一句多问
  - needs_term_mapping:    口语→条款术语映射需要

用法:
  python p3_annotate_questions.py
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

CANDIDATES_PATH = "intermediate/sampled_candidates.json"
OUTPUT_DIR = "intermediate"

# 粤语特征词
CANTONESE_MARKERS: list[str] = [
    "係", "咁", "嘅", "㗎", "喺", "哋", "嘢",
    "點解", "邊個", "點樣", "幾耐", "幾多",
    "冇", "唔", "佢", "乜", "嗰", "呢個",
    "點計", "點睇", "買嗰份", "份保單",
]

# 疑似错别字/谐音(核心保险术语)
TYPO_CANDIDATES: dict[str, str] = {
    # 常见错别字/谐音 → 正确写法
    "预存": "预缴(保证利息)",   # 可能是口语说法,不一定是错别字
    "守护挚保": "首护挚宝",     # 产品名常被写错
    "宏利红利": "宏利",         # 宏利被写成红利
    "买嗰份": "买那份",        # 粤语口语
}

# 条款术语映射提示(标注员判断)
TERM_MAPPING_HINTS: dict[str, str] = {
    "预存利率": "保证利息",
    "等候期": "等待期",
    "现金价值": "退保价值",
    "受保人": "被保险人",
    "保费回赠": "保费折扣",
    "供款": "缴费",
    "年期": "缴费年期",
    "红利率": "分红率",
}


# ============================================================================
# 自动检测
# ============================================================================

def auto_detect_lang(text: str) -> str:
    """自动检测语言。"""
    if not text:
        return "zh-Hans"

    # 粤语信号
    cantonese_count = sum(1 for m in CANTONESE_MARKERS if m in text)
    if cantonese_count >= 2:
        return "粤语夹杂"

    # 繁体信号
    trad_signals = ["爲", "麼", "後", "衞", "護", "險", "劃", "體", "醫", "療", "權"]
    trad_count = sum(1 for c in trad_signals if c in text)
    if trad_count >= 3:
        return "zh-Hant"

    return "zh-Hans"


def auto_detect_colloquial(text: str) -> bool:
    """检测是否口语/粤语表达。"""
    cantonese_count = sum(1 for m in CANTONESE_MARKERS if m in text)
    return cantonese_count >= 1


def auto_detect_typo_hints(text: str) -> list[dict]:
    """检测疑似错别字/谐音,返回提示列表。"""
    hints = []
    for wrong, correct in TYPO_CANDIDATES.items():
        if wrong in text:
            hints.append({
                "suspected_text": wrong,
                "likely_correct": correct,
                "confidence": "medium",  # 需人工确认
            })
    return hints


def auto_detect_term_mapping_hints(text: str) -> list[dict]:
    """检测可能需要术语映射的口语词。"""
    hints = []
    for spoken, formal in TERM_MAPPING_HINTS.items():
        if spoken in text:
            hints.append({
                "spoken_term": spoken,
                "formal_term": formal,
                "needs_human_confirm": True,
            })
    return hints


def auto_detect_underspecified(text: str) -> bool | None:
    """
    检测是否信息不全。
    规则推断(不确定,标 None 让人工判断):
      - 提到了产品名但没提年期
      - 问优惠/利率但没提时间段
    """
    # 这需要人工判断,自动检测只能给提示
    return None  # 不确定


def auto_detect_multi_intent(text: str) -> bool | None:
    """检测是否一句多问。"""
    # 多问号 / 多"吗" / 换行分隔多个问句
    question_marks = text.count("?") + text.count("？")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    line_questions = sum(1 for l in lines if "?" in l or "？" in l or l.endswith("吗"))
    if question_marks >= 2 or line_questions >= 2:
        return True
    # 逗号分隔的两个独立疑问
    if ("多少" in text or "怎么" in text) and ("可以" in text or "是否" in text):
        if text.count("？") >= 1 or text.count("?") >= 1:
            return True
    return None  # 不确定的留给人


# ============================================================================
# 主流程
# ============================================================================

def annotate_candidates(candidates: list[dict]) -> list[dict]:
    """为每条候选样本填充可自动检测的字段。"""
    annotated = []

    for c in candidates:
        question = c.get("question_raw", "")

        # 语言检测(可自动)
        if not c.get("question_lang") or c["question_lang"] == "zh-Hans":
            c["question_lang"] = auto_detect_lang(question)

        # question_type 从 session 取出
        if not c.get("question_type"):
            c["question_type"] = c.get("_p1_reason", "")

        # question_quality 自动检测
        qq = c.get("question_quality", {})

        # is_cross_company: 从 sample_type 推断
        if c.get("sample_type") == "multi_doc":
            qq["is_cross_company"] = True

        # is_colloquial
        if not qq.get("is_colloquial"):
            qq["is_colloquial"] = auto_detect_colloquial(question)

        # has_typo: 自动检测提示
        typo_hints = auto_detect_typo_hints(question)
        if typo_hints and not qq.get("has_typo"):
            qq["has_typo"] = True

        # 存储自动检测的提示供标注员参考
        c["_p3_auto_hints"] = {
            "lang": c["question_lang"],
            "cantonese_markers_found": [
                m for m in CANTONESE_MARKERS if m in question
            ],
            "typo_hints": typo_hints,
            "term_mapping_hints": auto_detect_term_mapping_hints(question),
            "possible_multi_intent": auto_detect_multi_intent(question),
            "possible_underspecified": auto_detect_underspecified(question),
        }

        # 保留 human-review 标记
        qq["_human_review_needed"] = True

        c["question_quality"] = qq
        annotated.append(c)

    return annotated


def generate_stats(annotated: list[dict]) -> dict:
    """生成标注统计。"""
    stats = {
        "total": len(annotated),
        "lang_distribution": Counter(),
        "colloquial_count": 0,
        "typo_count": 0,
        "term_mapping_hint_count": 0,
        "multi_intent_detected": 0,
        "cross_company_count": 0,
        "underspecified_hints": 0,
    }

    for c in annotated:
        stats["lang_distribution"][c.get("question_lang", "unknown")] += 1

        qq = c.get("question_quality", {})
        if qq.get("is_colloquial"):
            stats["colloquial_count"] += 1
        if qq.get("has_typo"):
            stats["typo_count"] += 1
        if qq.get("is_cross_company"):
            stats["cross_company_count"] += 1

        hints = c.get("_p3_auto_hints", {})
        if hints.get("term_mapping_hints"):
            stats["term_mapping_hint_count"] += 1
        if hints.get("possible_multi_intent"):
            stats["multi_intent_detected"] += 1
        if hints.get("possible_underspecified"):
            stats["underspecified_hints"] += 1

    return stats


def generate_annotation_guide(stats: dict) -> str:
    """生成标注指引 Markdown。"""
    return f"""# P3 问题质量标注指引

> 生成时间: {datetime.now().isoformat()}
> 待标注样本数: {stats['total']}

---

## 标注任务

对每条样本的 `question_quality` 子字段进行判定,共 7 个 bool 字段:

| 字段 | 判定标准 | 示例 |
|---|---|---|
| `is_underspecified` | 问题缺少关键限定(年期/计划名/币种),不补全就无法准确回答 | "宏挚传承保费多少"→true(没说年期) |
| `has_false_premise` | 问题含错误前提(如"agent说没等待期"但条款明确有) | "既然这个计划没有等待期..."→true |
| `is_multi_intent` | 一句包含2+个独立子问题,需分别回答 | "保额多少?保费怎么算?"→true |
| `is_cross_company` | 涉及多保司比较 | "宏利和友邦哪个好"→true |
| `is_colloquial` | 口语化/粤语表达('嘅'/'咁'/'係'等) | "呢个计划点样计保费㗎"→true |
| `has_typo` | 含错别字/谐音 | "宏利"写成"红利"→true |
| `needs_term_mapping` | 含口语词需映射到条款术语,且这是RAG检索失败的关键原因 | "预存利率"→"保证利息"→true |

## 术语映射参考

以下口语→术语的映射已知会导致 RAG 检索失败,如样本含这些词,`needs_term_mapping` 标 true:

| 口语/用户用词 | 条款正式术语 |
|---|---|
| 预存利率 | 保证利息 |
| 等候期 | 等待期 |
| 受保人 | 被保险人 |
| 保费回赠 | 保费折扣 |
| 供款 | 缴费 |
| 红利率 | 分红率 |

## 自动检测提示

脚本已自动检测并标记以下字段,标注员只需复核:

- `question_lang`: 自动检测简体/繁体/粤语
- `is_colloquial`: 自动检测粤语特征词
- `has_typo`: 自动检测常见错别字(confidence=medium,需人工确认)
- `is_cross_company`: 从 sample_type=multi_doc 自动推断

## 统计概览

| 指标 | 值 |
|---|---|
| 待标注总数 | {stats['total']} |
| 自动检测口语 | {stats['colloquial_count']} |
| 自动检测错别字 | {stats['typo_count']} |
| 跨保司题 | {stats['cross_company_count']} |
| 术语映射提示 | {stats['term_mapping_hint_count']} |
| 疑似多意图 | {stats['multi_intent_detected']} |
"""


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="P3: Question Quality Annotation")
    parser.add_argument("--candidates", type=str, default=CANDIDATES_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载候选集
    print(f"加载候选集: {args.candidates}")
    with open(args.candidates, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"  共 {len(candidates)} 条")

    # 自动标注
    print("执行自动检测...")
    annotated = annotate_candidates(candidates)

    # 生成统计
    stats = generate_stats(annotated)
    print(f"  自动检测完成:")
    print(f"    粤语/口语: {stats['colloquial_count']}")
    print(f"    疑似错别字: {stats['typo_count']}")
    print(f"    术语映射提示: {stats['term_mapping_hint_count']}")
    print(f"    疑似多意图: {stats['multi_intent_detected']}")

    # 写入标注结果
    out_path = os.path.join(args.output_dir, "question_annotated.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, ensure_ascii=False, indent=2)
    print(f"\n写入标注结果: {out_path}")

    # 写入标注指引
    guide_path = os.path.join(args.output_dir, "p3_annotation_guide.md")
    guide = generate_annotation_guide(stats)
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide)
    print(f"写入标注指引: {guide_path}")

    # 写入统计
    summary_path = os.path.join(args.output_dir, "p3_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "stats": {
                "total": stats["total"],
                "lang_distribution": dict(stats["lang_distribution"]),
                "colloquial_count": stats["colloquial_count"],
                "typo_count": stats["typo_count"],
                "term_mapping_hint_count": stats["term_mapping_hint_count"],
                "multi_intent_detected": stats["multi_intent_detected"],
                "cross_company_count": stats["cross_company_count"],
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"写入统计:    {summary_path}")

    print("\nDone. 标注员请打开 p3_annotation_guide.md 阅读指引.")


if __name__ == "__main__":
    main()
