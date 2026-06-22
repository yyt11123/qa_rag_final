"""
P4: 检索核实框架 — Retrieval Ground Truth
===========================================
为每条候选样本构造 gold_evidence,核验 no_answer_in_corpus。

核心原则:
  gold_evidence 一律走"当前库重查+人工选"。
  不反查 hash——DashVector 返回的 metadata 自带 hash/page/type/node_path/file_name。
  只依赖 DashVector,不依赖中台 API。

输入:
  - intermediate/question_annotated.json  (P3 产出)

输出:
  - intermediate/p4_verification_batch.json  — 待人工逐条核实的工作单
  - intermediate/retrieval_verified.json     — 核实后的候选集
  - intermediate/p4_summary.json             — 核实统计

工作流程:

  对每条样本:
    1. 用 question_raw 查询 DashVector(需要 DAHSVECTOR_API_KEY 环境变量)
    2. 拉取 top-K=15 结果
    3. 生成"人工核实工作单":展示 question_raw + topK 结果摘要
    4. 标注员逐条判断:
       a. 哪些 chunk 真正包含答案 → 勾选进 gold_evidence
       b. 哪些高分但不相关 → 标记进 irrelevant_high_score(供 P5 提炼 forbidden)
    5. 判断 no_answer_in_corpus / requires_multi_doc
    6. 对 suspected_refusal 样本:翻转或确认

用法:
  python p4_retrieval_verify.py --mode prepare    # 生成人工核实工作单
  python p4_retrieval_verify.py --mode apply      # 应用人工核实结果
  python p4_retrieval_verify.py --mode dry-run    # 不连 DashVector,用检索快照模拟
"""

import json
import os
import sys
import argparse
from collections import Counter
from datetime import datetime


# ============================================================================
# 配置
# ============================================================================

ANNOTATED_PATH = "intermediate/question_annotated.json"
OUTPUT_DIR = "intermediate"
TOP_K = 15


# ============================================================================
# DashVector 查询(需要环境变量)
# ============================================================================

def query_dashvector(question: str, company: str | None, top_k: int = TOP_K) -> list[dict]:
    """
    查询 DashVector 获取候选 chunk。
    需要环境变量: DASHSVECTOR_API_KEY, DASHSVECTOR_ENDPOINT, DASHSVECTOR_COLLECTION
    """
    api_key = os.environ.get("DASHSVECTOR_API_KEY")
    endpoint = os.environ.get("DASHSVECTOR_ENDPOINT")
    collection = os.environ.get("DASHSVECTOR_COLLECTION", "awm_docs")

    if not api_key or not endpoint:
        raise RuntimeError(
            "DashVector 连接信息缺失。请设置环境变量:\n"
            "  DASHSVECTOR_API_KEY=xxx\n"
            "  DASHSVECTOR_ENDPOINT=xxx\n"
            "  DASHSVECTOR_COLLECTION=awm_docs  (可选)"
        )

    # 构建 filter
    filter_expr = None
    if company:
        filter_expr = f'company == "{company}"'

    # 调用 DashVector API(伪代码,需根据实际 SDK 调整)
    # from dashvector import Client
    # client = Client(api_key=api_key, endpoint=endpoint)
    # coll = client.get_collection(collection)
    # results = coll.query(
    #     vector=embed(question),
    #     top_k=top_k,
    #     filter=filter_expr,
    #     output_fields=["hash", "page", "type", "node_path", "file_name",
    #                    "company", "product_name", "kind", "chunk_type"],
    # )
    #
    # return [
    #     {
    #         "hash": doc.get("hash", ""),
    #         "page": doc.get("page", 1),
    #         "doc_type": doc.get("type", ""),
    #         "node_path": doc.get("node_path", ""),
    #         "file_name": doc.get("file_name", ""),
    #         "company": doc.get("company", ""),
    #         "product_name": doc.get("product_name", ""),
    #         "kind": doc.get("kind", ""),
    #         "chunk_type": doc.get("chunk_type", ""),
    #         "score": doc.get("score", 0.0),
    #         "text_preview": doc.get("text", "")[:500],
    #     }
    #     for doc in results
    # ]

    # 暂时返回占位,实际运行时需替换
    raise NotImplementedError(
        "DashVector 查询接口未实现。请根据实际 DashVector SDK 完成 query_dashvector() 函数。"
    )


# ============================================================================
# 核实工作单生成
# ============================================================================

def generate_verification_batch(
    candidates: list[dict],
    dry_run: bool = False,
) -> list[dict]:
    """
    为每条候选样本生成待人工核实的工作单。

    每条工作单包含:
      - 样本基本信息(id, question_raw, sample_type)
      - topK 检索结果摘要(需标注员逐条判定)
      - 标注选项(gold_evidence / irrelevant / 不确定)
    """
    batch = []

    for c in candidates:
        question = c.get("question_raw", "")
        company = c.get("source_company")

        # 查询 DashVector(或 dry-run)
        if dry_run:
            # 用已有的 retrieval_candidates_snapshot 模拟
            topk = c.get("retrieval_candidates_snapshot", [])
        else:
            try:
                topk = query_dashvector(question, company)
            except NotImplementedError:
                topk = []

        work_item = {
            "_work_id": c["id"],
            "_status": "pending",  # pending → in_progress → verified
            "_annotator": "",
            "candidate_id": c["id"],
            "question_raw": question,
            "sample_type": c["sample_type"],
            "source_company": company,
            "product_name": c.get("product_name"),
            "original_system_refusal": c["sample_type"] == "correct_refusal",

            # 检索结果(标注员逐条判定)
            "_retrieval_results": [
                {
                    "_chunk_index": i,
                    "_verdict": None,  # 标注员填: "gold" / "irrelevant" / "uncertain"
                    "_verdict_note": "",
                    "hash": doc.get("hash", ""),
                    "page": doc.get("page", 1),
                    "doc_type": doc.get("doc_type", ""),
                    "node_path": doc.get("node_path", ""),
                    "file_name": doc.get("file_name", ""),
                    "score": doc.get("score", 0.0),
                    "text_preview": doc.get("text_preview", doc.get("text", ""))[:500],
                    "company": doc.get("company", ""),
                    "product_name": doc.get("product_name", ""),
                    "kind": doc.get("kind", ""),
                }
                for i, doc in enumerate(topk)
            ],

            # 核验结论(标注员最终填)
            "_verification_result": {
                "no_answer_in_corpus": None,     # True / False
                "requires_multi_doc": c.get("requires_multi_doc", False),
                "gold_evidence": [],               # 标注员选中的 chunk 汇总
                "irrelevant_high_score": [],       # 高分但不相关的(供P5)
                "sample_type_flipped_to": None,    # 如翻转: "normal_answer" / "correct_refusal"
                "flip_reason": "",
                "notes": "",
            },
        }

        batch.append(work_item)

    return batch


def apply_verification(
    candidates: list[dict],
    batch: list[dict],
) -> list[dict]:
    """将人工核实结果应用到候选集。"""
    # 建索引
    batch_index = {b["candidate_id"]: b for b in batch}

    verified = []
    for c in candidates:
        work = batch_index.get(c["id"])
        if not work:
            verified.append(c)
            continue

        vr = work.get("_verification_result", {})

        # gold_evidence
        gold_docs = vr.get("gold_evidence", [])
        if gold_docs:
            c["gold_evidence"] = [
                {
                    "hash": g.get("hash", ""),
                    "page": g.get("page", 1),
                    "doc_type": g.get("doc_type", ""),
                    "node_path": g.get("node_path", ""),
                    "file_name": g.get("file_name", ""),
                }
                for g in gold_docs
            ]

        # no_answer_in_corpus
        no_answer = vr.get("no_answer_in_corpus")
        if no_answer is not None:
            c["no_answer_in_corpus"] = no_answer

        # requires_multi_doc
        c["requires_multi_doc"] = vr.get("requires_multi_doc", c.get("requires_multi_doc", False))

        # retrieval snapshot 存档
        c["retrieval_candidates_snapshot"] = [
            {
                "hash": r.get("hash", ""),
                "page": r.get("page", 1),
                "file_name": r.get("file_name", ""),
                "score": r.get("score", 0.0),
                "kind": r.get("kind", ""),
            }
            for r in work.get("_retrieval_results", [])
        ]

        # sample_type 翻转
        flip_to = vr.get("sample_type_flipped_to")
        if flip_to:
            old_type = c["sample_type"]
            c["sample_type"] = flip_to
            c["notes"] = (
                f"[P4翻转] {old_type} → {flip_to}: "
                f"{vr.get('flip_reason', '')}" +
                ("\n" + c.get("notes", "") if c.get("notes") else "")
            )

            # 翻转后调整 eval_method / expected_behavior
            if flip_to == "normal_answer":
                c["expected_behavior"] = "answer"
                c["eval_method"] = "objective_qa"
            elif flip_to == "correct_refusal":
                c["expected_behavior"] = "refuse"
                c["eval_method"] = "refusal"
                c["base_answer"] = ""

        # irrelevant_high_score(供 P5 用)
        irrelevant = vr.get("irrelevant_high_score", [])
        if irrelevant:
            c["_p5_forbidden_hints"] = irrelevant

        c["kb_version_for_evidence"] = "rag_v1_260604_bge_m3"
        verified.append(c)

    return verified


def print_summary(verified: list[dict]) -> None:
    """打印核实统计。"""
    print("=" * 62)
    print("  P4 检索核实 — 汇总统计")
    print("=" * 62)

    total = len(verified)
    has_gold = sum(1 for c in verified if c.get("gold_evidence"))
    no_answer = sum(1 for c in verified if c.get("no_answer_in_corpus"))
    multi_doc = sum(1 for c in verified if c.get("requires_multi_doc"))

    print(f"  总样本数:              {total}")
    print(f"  有 gold_evidence:      {has_gold}")
    print(f"  no_answer_in_corpus:   {no_answer}")
    print(f"  requires_multi_doc:    {multi_doc}")

    # sample_type 翻转统计
    flips = [c for c in verified if "[P4翻转]" in (c.get("notes") or "")]
    print(f"  sample_type 翻转:       {len(flips)}")
    if flips:
        for f in flips:
            original = f["notes"].split("→")[0].split("] ")[-1] if "→" in f["notes"] else "?"
            new = f["sample_type"]
            print(f"    {f['id']}: {original} → {new}")

    print("=" * 62)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="P4: Retrieval Verification")
    parser.add_argument("--mode", choices=["prepare", "apply", "dry-run"],
                        default="dry-run",
                        help="prepare:生成工作单 | apply:应用人工结果 | dry-run:模拟(默认)")
    parser.add_argument("--annotated", type=str, default=ANNOTATED_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--batch-file", type=str,
                        default="intermediate/p4_verification_batch.json",
                        help="人工核实工作单路径")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"加载: {args.annotated}")
    with open(args.annotated, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"  共 {len(candidates)} 条")

    if args.mode in ("prepare", "dry-run"):
        is_dry = args.mode == "dry-run"
        if is_dry:
            print("  ⚠ dry-run 模式:不连 DashVector,用 retrieval_candidates_snapshot 模拟")

        batch = generate_verification_batch(candidates, dry_run=is_dry)

        with open(args.batch_file, "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
        print(f"\n写入工作单: {args.batch_file} ({len(batch)} 条)")
        print("标注员请逐条填写 _retrieval_results[]._verdict 和 _verification_result")

    elif args.mode == "apply":
        print(f"加载人工核实结果: {args.batch_file}")
        with open(args.batch_file, "r", encoding="utf-8") as f:
            batch = json.load(f)

        verified = apply_verification(candidates, batch)
        print_summary(verified)

        out_path = os.path.join(args.output_dir, "retrieval_verified.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(verified, f, ensure_ascii=False, indent=2)
        print(f"\n写入核实结果: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
