"""
P0: 对话切分 — Session → Question Units
========================================
把企微多轮对话拆分为独立的"问题单元"(question unit)。

核心原则:
  P0 负责"把散的拼起来"，不负责"把缺的补上"。
  question_raw 保留用户原话拼接(不补全、不消解指代、不改写),
  上下文推断结果存入 context_company/context_product 独立字段。

输入: real_conversation_recording/sessions/*.json
输出:
  1. intermediate/question_units.json      — 所有切分出的 question unit
  2. intermediate/p0_summary.json          — 汇总统计
  3. intermediate/p0_llm_pending.json      — 需要 LLM 做上下文推断的 unit(可选步骤)
  4. intermediate/p0_errors.json           — 处理异常的 session

切分规则(纯规则,不调 LLM):
  ─────────────────────────────────────────────────
  验收标准 / 边界情形处理:

  AC1  消息按 created_at 排序,不假设原始数组有序。
       同时间戳时 user 排在 assistant 前,同时间戳同角色维持原序。

  AC2  以 assistant 回复为边界切分 turn:
       连续的 user 消息 → 聚合成一个 question_raw,
       遇到 assistant 回复 → 该 turn 结束,
       assistant 之后的 user 消息 → 新 turn 开始。

  AC3  assistant 主动开口(第一条就是 assistant):跳过,不产出空 unit。

  AC4  连续 user 消息(未等 assistant 回复就发下一条):
       合并到同一个 unit。例:
         user: "在吗"
         user: "我想问下宏利等待期"
         → question_raw = "在吗\n我想问下宏利等待期"

  AC5  一个 unit 内用户问了多件事(如"等待期几耐?保费点计?"):
       P0 不拆分——这是 P3 is_multi_intent 的事。
       P0 切分单位 = "一次提问回合",不是"一个原子问题"。

  AC6  跨 assistant 回复的追问检测:
       如果新 unit 的 question_raw 中不含保司名/产品名,
       且同一 session 的前一 unit 含有这些信息,
       → 标记 is_context_inferred=true,
       上下文从上一 unit 或上一 assistant 回复的 metadata 继承。

  AC7  greeting/寒暄过滤:
       如果 unit 所有 user 消息合并后只是问候/寒暄/测试,
       标记 is_substantive=false,不进入后续标注流程。

上下文推断(rule-based 优先,LLM 仅兜底):
  ─────────────────────────────────────────────────
  大部分情况规则能搞定:当前 unit 无产品名 → 从上一 unit 继承。
  需 LLM 介入的少数情况:同一 session 里讨论过多款产品,
  无法确定追问指向哪款时,标记 pending 等 LLM 处理。

用法:
  python p0_session_to_units.py                        # 完整流程
  python p0_session_to_units.py --no-llm               # 跳过 LLM,规则覆盖的先标
  python p0_session_to_units.py --session-id 144263...  # 单 session 调试
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
OUTPUT_DIR = "intermediate"
KB_VERSION = "rag_v1_260604_bge_m3"

# 保司短名 → 标准名映射(19 家,从 metadata 和中台对齐)
COMPANY_ALIASES: dict[str, str] = {
    "友邦": "友邦保险", "友邦保险": "友邦保险", "AIA": "友邦保险", "aia": "友邦保险",
    "宏利": "宏利保险", "宏利保险": "宏利保险", "Manulife": "宏利保险",
    "保诚": "保诚保险", "保诚保险": "保诚保险", "Prudential": "保诚保险",
    "安盛": "安盛保险", "安盛保险": "安盛保险", "AXA": "安盛保险",
    "万通": "万通保险", "万通保险": "万通保险",
    "富卫": "富卫保险", "富卫保险": "富卫保险", "FWD": "富卫保险",
    "周大福": "周大福人寿", "周大福人寿": "周大福人寿",
    "中国人寿": "中国人寿", "国寿": "中国人寿",
    "中国太平": "中国太平", "太平": "中国太平",
    "中银": "中银人寿", "中银人寿": "中银人寿",
    "汇丰": "汇丰人寿", "汇丰人寿": "汇丰人寿", "HSBC": "汇丰人寿",
    "恒生": "恒生保险", "恒生保险": "恒生保险",
    "安达": "安达人寿", "安达人寿": "安达人寿",
    "永明": "永明金融", "永明金融": "永明金融", "SunLife": "永明金融",
    "苏黎世": "苏黎世保险", "苏黎世保险": "苏黎世保险",
    "立桥": "立桥人寿", "立桥人寿": "立桥人寿",
    "富邦": "富邦保险", "富邦保险": "富邦保险",
    "大都会": "大都会人寿", "大都会人寿": "大都会人寿",
    "忠意": "忠意保险", "忠意保险": "忠意保险",
    "东亚": "东亚保险", "东亚保险": "东亚保险",
}

# 纯寒暄/测试 特征
GREETING_PATTERNS: list[str] = [
    r"^(你好|您好|hi|hello|嗨|在吗|在不在|有人吗)[\s!！。.,，~～]*$",
    r"^(早上好|下午好|晚上好|晚安|早安|午安)[\s!！。.,，~～]*$",
    r"^(谢谢|多谢|thank|thanks|ok|好的|明白了|收到|了解)[\s!！。.,，~～]*$",
    r"^(再见|拜拜|bye|88)[\s!！。.,，~～]*$",
    r"^(What.s your name|Repeat your last|test|测试).*",
    r"^@chatbot\s*(What.s your name|Repeat your last|test|测试).*",
]

# assistant 回答中用于提取上下文的字段路径
CONTEXT_SIGNALS_FROM_ANSWER: list[str] = [
    "metadata.company_name",       # assistant metadata 里有 company_name 数组
    "metadata.nodes_history..result.extracted_info.insurance_companies",
]


# ============================================================================
# 工具函数
# ============================================================================

def sort_messages(messages: list[dict]) -> list[dict]:
    """按 created_at 排序,同时间戳维持原始交替顺序。

    关键:不能按 role_priority 排——企微数据里同一回合的 user/assistant
    共享 created_at,按 role_priority 会把所有 user 聚到一起、所有 assistant
    聚到一起,破坏 user→assistant→user→assistant 的交替结构。
    """
    def sort_key(idx_msg: tuple[int, dict]) -> tuple:
        idx, m = idx_msg
        ts = m.get("created_at", 0)
        # 同时间戳用原始 idx 保序,不按 role 排序
        return (ts, idx)

    indexed = list(enumerate(messages))
    indexed.sort(key=sort_key)
    return [m for _, m in indexed]


def detect_company_in_text(text: str) -> list[str]:
    """从文本中检测提及的保司名,返回标准名列表。"""
    found = []
    for alias, standard in COMPANY_ALIASES.items():
        if alias in text:
            if standard not in found:
                found.append(standard)
    return found


def detect_product_in_answer_metadata(assistant_msg: dict) -> str | None:
    """从 assistant 消息的 metadata 中提取产品名。"""
    meta = assistant_msg.get("metadata", {})
    if not isinstance(meta, dict):
        return None

    # 从 nodes_history 中找产品相关信息
    nodes = meta.get("nodes_history", [])
    if not isinstance(nodes, list):
        return None

    for node in nodes:
        if not isinstance(node, dict):
            continue
        result = node.get("result", {})
        # result 可能是 dict(如 simpleQA_response_node) 或 list(如 docs_relevance_check_node)
        if not isinstance(result, dict):
            continue

        # extracted_info 中的 product_name
        extracted = result.get("extracted_info", {})
        if isinstance(extracted, dict):
            pn = extracted.get("product_name")
            if pn:
                return pn

        # topk_docs 中的 product_name
        topk = result.get("topk_docs", [])
        if isinstance(topk, list):
            for doc in topk:
                if not isinstance(doc, dict):
                    continue
                pn = doc.get("product_name", "")
                if pn and pn.strip():
                    return pn.strip()
    return None


def detect_company_in_answer_metadata(assistant_msg: dict) -> list[str]:
    """从 assistant 消息的 metadata 中提取保司名。"""
    companies = []
    meta = assistant_msg.get("metadata", {})
    # 直接字段
    cn = meta.get("company_name", [])
    if isinstance(cn, list):
        companies.extend(cn)
    elif isinstance(cn, str) and cn:
        companies.append(cn)
    return [c for c in companies if c]


def is_greeting(text: str) -> bool:
    """判断合并后的用户消息是否为纯寒暄/测试。

    用正信号(是否包含保险/保司内容)而非长度阈值——
    避免'立桥可以爷爷给孙子买吗'被误判为greeting。
    """
    clean = text.strip().lower()
    # 去掉 @chatbot 前缀
    clean = re.sub(r'^@chatbot\s*', '', clean)

    if not clean:
        return True

    # 模板匹配:明确的寒暄/测试/重复询问名字
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, clean, re.IGNORECASE):
            return True

    # 正信号:包含任何保司名/保险术语/产品相关词 → 不是寒暄
    insurance_terms = [
        # 核心保险词
        "保费", "保额", "保障", "保单", "保险", "理赔", "退保",
        "等待期", "受保人", "投保人", "受益人", "持有人", "保单持有人",
        "豁免", "危疾", "重疾", "医疗", "储蓄", "年金", "分红",
        "预缴", "趸交", "年期", "缴付", "保费缴付",
        # 优惠/产品相关
        "优惠", "折扣", "利率", "条款", "缴费", "起售", "投保",
        "产品", "计划", "多元化", "多元货币",
        # 功能/操作相关
        "变更", "设置", "选项", "继承", "隔代", "代投保",
        "索赔", "身故", "赔付", "赔偿",
        # 单字(保险核心)
        "保", "险", "赔", "缴",
        # 币种
        "美元", "港元", "港币", "人民币",
        # 问句特征(包含实质性疑问)
        "多少", "怎么", "如何", "可以", "能否", "是否",
        "什么", "哪些", "哪家", "哪个", "有没有",
        "要求", "条件", "规定", "限制", "区别", "对比",
    ]
    has_insurance_term = any(term in clean for term in insurance_terms)

    # 正信号:包含任何保司名
    has_company = bool(detect_company_in_text(clean))

    # 正信号:包含常见产品名模式
    product_patterns = [
        r'[a-zA-Z]{2,}',           # 英文缩写(AIA/FWD等)
        r'\d+年',                   # X年缴/年期
        r'[0-9]+(万|千|百|%|％)',   # 金额/比例
        r'(计划|保|系列|版)',        # 产品名后缀
    ]
    has_product_like = any(re.search(p, clean) for p in product_patterns)

    if has_insurance_term or has_company or has_product_like:
        return False

    return True


def extract_context_from_unit(unit: dict) -> tuple[str | None, str | None]:
    """从 unit 中提取明确的保司名和产品名(规则方式)。"""
    question = unit.get("question_raw", "")
    companies = detect_company_in_text(question)

    # 如果有 session metadata,也看看
    session_companies = unit.get("_session_company_names", [])
    if not companies and session_companies:
        companies = list(session_companies)

    company = companies[0] if len(companies) == 1 else (companies if companies else None)
    if isinstance(company, list):
        company = None  # 多个公司,不确定

    # 产品名:目前只能从 assistant metadata 提取
    product = unit.get("_assistant_product")

    return company, product


def _has_followup_signal(question_raw: str) -> bool:
    """检测问题是否包含'追问前文'的信号——只有真 follow-up 才继承上下文。

    核心原则:问题能否脱离前文独立成立?不能 → 需要上下文继承。
    """
    # 指代词(明确指向前文)
    demonstratives = [
        "这个", "那个", "这款", "那款", "这份", "那份",
        "该计划", "该产品", "该保单", "该公司",
        "它", "其", "此", "上述", "以上",
        "前面", "刚才", "刚刚", "上次", "上一个",
        "上面那个", "之前那个", "前文", "前述",
    ]
    if any(d in question_raw for d in demonstratives):
        return True

    # 明确引用前一轮 assistant 回答
    ref_patterns = [
        r"回答有问题",
        r"你(刚才|刚刚|上面|前面).{0,5}(说|提|讲|写|解释|答)",
        r"(再|继续).{0,5}(说|讲|解释|详细|问)",
        r"(具体|详细).{0,3}(说|讲|解释|一下|一点)",
    ]
    for pat in ref_patterns:
        if re.search(pat, question_raw):
            return True

    # 追问开头(句首的转折/承接词)
    followup_starters = [
        r"^(那|那么|还有|另外|然后|所以)",
    ]
    for pat in followup_starters:
        if re.search(pat, question_raw):
            return True

    # 短 fragment + 语气词 → 大概率是追问(如"优惠政策呢")
    # 但要排除:含有新保司名/产品名的独立短问题
    if len(question_raw) <= 15:
        if re.search(r"[呢吧]$", question_raw):
            # 如果这个短 fragment 里有新保司名,它是独立的新问题
            if detect_company_in_text(question_raw):
                return False
            return True

    return False


def rule_based_context_inherit(
    unit: dict,
    prev_unit: dict | None,
) -> dict:
    """
    规则方式推断上下文。
    只在检测到追问信号时才从上一 unit 继承;
    独立新问题即使无保司/产品名也不盲传。
    """
    if prev_unit is None:
        return unit

    question = unit.get("question_raw", "")
    company, product = extract_context_from_unit(unit)

    # 只在对前文有明确指代/追问时继承
    if not _has_followup_signal(question):
        # 无追问信号 → 不继承,但保留 unit 自身能检测到的信息
        unit["context_company"] = company
        unit["context_product"] = product
        return unit

    # 有追问信号 → 继承前文
    if company is None:
        prev_company = prev_unit.get("context_company")
        if prev_company and isinstance(prev_company, str):
            company = prev_company
            unit["is_context_inferred"] = True
            unit["inference_basis"] = (
                f"保司名从 unit {prev_unit['unit_id']} 的 context_company 继承"
            )

    if product is None:
        prev_product = prev_unit.get("context_product")
        if not prev_product:
            prev_product = prev_unit.get("_assistant_product")
        if prev_product:
            product = prev_product
            unit["is_context_inferred"] = True
            existing = (unit.get("inference_basis") or "")
            if existing:
                existing += "; "
            unit["inference_basis"] = existing + (
                f"产品名从 unit {prev_unit['unit_id']} 继承"
            )

    unit["context_company"] = company
    unit["context_product"] = product

    return unit


def build_question_units(session: dict) -> list[dict]:
    """
    核心切分逻辑:将一个 session 的 messages 切分为 question units。

    算法:
      1. 排序 messages
      2. 顺序扫描,以 assistant 回复为边界
      3. 连续的 user 消息合并
      4. 遇到 assistant → 产出 unit(如有 user 消息)
      5. greeting 过滤
    """
    s = session.get("session", {})
    session_id = s.get("id", "unknown")
    msgs = session.get("messages", [])

    if not msgs:
        return []

    # Step 1: 排序
    sorted_msgs = sort_messages(msgs)

    # 从 session metadata 提取全局信息
    session_companies = s.get("company_names", [])
    session_products = s.get("product_names", [])

    # Step 2: 按 assistant 边界切分
    units = []
    current_user_msgs: list[dict] = []
    unit_seq = 0
    prev_assistant_msg: dict | None = None

    for msg in sorted_msgs:
        role = msg.get("role", "")

        if role == "user":
            current_user_msgs.append(msg)
        elif role == "assistant":
            # assistant 回复作为 turn 边界
            if current_user_msgs:
                unit_seq += 1
                unit = _create_unit(
                    session_id=session_id,
                    seq=unit_seq,
                    user_msgs=current_user_msgs,
                    assistant_msg=msg,
                    session_companies=session_companies,
                    session_products=session_products,
                )
                units.append(unit)
                prev_assistant_msg = msg
                current_user_msgs = []
            else:
                # AC3: assistant 先开口,前面没有 user 消息 → 跳过
                pass
            # 更新 prev_assistant 用于下一轮上下文推断
            prev_assistant_msg = msg

    # 末尾可能残留 user 消息,无 assistant 回复
    if current_user_msgs:
        unit_seq += 1
        unit = _create_unit(
            session_id=session_id,
            seq=unit_seq,
            user_msgs=current_user_msgs,
            assistant_msg=None,  # 无回复
            session_companies=session_companies,
            session_products=session_products,
        )
        units.append(unit)

    # Step 3: 跨 turn 上下文继承(rule-based)
    for i in range(1, len(units)):
        units[i] = rule_based_context_inherit(units[i], units[i - 1])

    # 第一个 unit 如果没有推断上下文,也从 session metadata 补
    if units:
        if not units[0].get("context_company"):
            if session_companies:
                units[0]["context_company"] = session_companies[0]
        if not units[0].get("context_product"):
            if session_products:
                units[0]["context_product"] = session_products[0]

    return units


def _create_unit(
    session_id: str,
    seq: int,
    user_msgs: list[dict],
    assistant_msg: dict | None,
    session_companies: list[str],
    session_products: list[str],
) -> dict:
    """从一组连续 user 消息 + 一条 assistant 回复,构建一个 question unit。"""

    # question_raw: 合并用户原话(用换行拼接)
    question_parts = [m.get("content", "").strip() for m in user_msgs]
    question_raw = "\n".join(p for p in question_parts if p)

    # source_message_ids
    source_ids = [m.get("id", "") for m in user_msgs if m.get("id")]

    # 从 question_raw 检测保司/产品
    companies_in_q = detect_company_in_text(question_raw)

    # 从 assistant metadata 提取
    product_from_asst = None
    companies_from_asst: list[str] = []
    if assistant_msg:
        product_from_asst = detect_product_in_answer_metadata(assistant_msg)
        companies_from_asst = detect_company_in_answer_metadata(assistant_msg)

    # 综合确定 context_company
    context_company = None
    if len(companies_in_q) == 1:
        context_company = companies_in_q[0]
    elif companies_from_asst:
        context_company = companies_from_asst[0]
    elif len(session_companies) == 1:
        context_company = session_companies[0]

    # 综合确定 context_product
    context_product = None
    if product_from_asst:
        context_product = product_from_asst
    elif session_products:
        context_product = session_products[0]

    # greeting 判定
    substantive = not is_greeting(question_raw)

    unit = {
        "unit_id": f"{session_id}-{seq}",
        "session_id": session_id,
        "question_raw": question_raw,
        "source_message_ids": source_ids,
        "is_substantive": substantive,
        "context_company": context_company,
        "context_product": context_product,
        "is_context_inferred": False,
        "inference_basis": None,
        # 以下为内部字段(不进评测集,供后续步骤使用)
        "_session_company_names": session_companies,
        "_session_product_names": session_products,
        "_assistant_product": product_from_asst,
        "_assistant_has_response": assistant_msg is not None,
        "_assistant_content_preview": (
            assistant_msg.get("content", "")[:200] if assistant_msg else None
        ),
        "_turn_index": seq,
    }

    return unit


# ============================================================================
# LLM 上下文推断(可选步骤)
# ============================================================================

def detect_units_needing_llm(units: list[dict]) -> list[dict]:
    """
    找出规则无法确定上下文、需要 LLM 介入的 unit。

    条件:
      1. is_substantive=true
      2. context_company 为空 或 为列表(多保司不确定)
      3. 且同一 session 的前序 unit 有多种可能
    """
    pending = []
    for unit in units:
        if not unit["is_substantive"]:
            continue
        company = unit.get("context_company")
        # 完全没有保司信息,或保司有多家不确定
        if company is None and unit.get("is_context_inferred"):
            # 规则已经尝试继承但失败了
            pending.append(unit)
    return pending


# ============================================================================
# 主流程
# ============================================================================

def load_sessions(sessions_dir: str) -> list[dict]:
    """加载所有 session JSON 文件。"""
    pattern = os.path.join(sessions_dir, "*.json")
    files = sorted(glob.glob(pattern))
    sessions = []
    errors = []

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                session = json.load(f)
            sessions.append(session)
        except json.JSONDecodeError as e:
            errors.append({"file": fpath, "error": str(e)})
        except Exception as e:
            errors.append({"file": fpath, "error": str(e)})

    return sessions, errors


def process_all_sessions(sessions: list[dict]) -> dict:
    """处理所有 session,返回全部 question units + 统计信息。"""
    all_units: list[dict] = []
    stats = {
        "total_sessions": len(sessions),
        "empty_sessions": 0,
        "single_turn_sessions": 0,
        "multi_turn_sessions": 0,
        "total_units": 0,
        "substantive_units": 0,
        "non_substantive_units": 0,
        "context_inferred_units": 0,
        "llm_pending_units": 0,
        "company_distribution": Counter(),
        "units_per_session_distribution": Counter(),
    }
    errors = []

    for session in sessions:
        s = session.get("session", {})
        session_id = s.get("id", "unknown")
        msgs = session.get("messages", [])

        if not msgs:
            stats["empty_sessions"] += 1
            continue

        try:
            units = build_question_units(session)
        except Exception as e:
            errors.append({"session_id": session_id, "error": str(e)})
            continue

        # 统计
        n_units = len(units)
        stats["total_units"] += n_units
        stats["units_per_session_distribution"][n_units] += 1

        if n_units <= 1:
            stats["single_turn_sessions"] += 1
        else:
            stats["multi_turn_sessions"] += 1

        for unit in units:
            all_units.append(unit)
            if unit["is_substantive"]:
                stats["substantive_units"] += 1
                company = unit.get("context_company")
                if company and isinstance(company, str):
                    stats["company_distribution"][company] += 1
            else:
                stats["non_substantive_units"] += 1

            if unit.get("is_context_inferred"):
                stats["context_inferred_units"] += 1

    # 检测需要 LLM 的 unit
    llm_pending = detect_units_needing_llm(all_units)
    stats["llm_pending_units"] = len(llm_pending)

    return {
        "units": all_units,
        "stats": stats,
        "llm_pending": llm_pending,
        "errors": errors,
    }


def print_summary(result: dict) -> None:
    """打印汇总统计到控制台。"""
    stats = result["stats"]
    total = stats["total_sessions"]

    print("=" * 62)
    print("  P0 对话切分 — 汇总统计")
    print("=" * 62)
    print(f"  总 session 数:           {total:>6}")
    print(f"  空 session(0条消息):     {stats['empty_sessions']:>6}")
    print(f"  单轮 session(≤1 unit):   {stats['single_turn_sessions']:>6}")
    print(f"  多轮 session(>1 unit):   {stats['multi_turn_sessions']:>6}")
    print(f"  ─────────────────────────────")
    print(f"  总 question unit 数:     {stats['total_units']:>6}")
    print(f"  实质性 unit:             {stats['substantive_units']:>6}")
    print(f"  非实质性 unit(greeting):  {stats['non_substantive_units']:>6}")
    print(f"  上下文推断 unit:         {stats['context_inferred_units']:>6}")
    print(f"  待 LLM 处理 unit:        {stats['llm_pending_units']:>6}")
    print(f"  处理异常 session:        {len(result['errors']):>6}")

    if total > 0:
        avg = stats['total_units'] / total
        print(f"  ─────────────────────────────")
        print(f"  平均每 session unit 数:   {avg:.2f}")
        sub_rate = stats['substantive_units'] / max(stats['total_units'], 1) * 100
        print(f"  实质性 unit 占比:         {sub_rate:.1f}%")

    print()
    print("  Unit 数分布 (top 15):")
    dist = stats["units_per_session_distribution"]
    for n, c in sorted(dist.items())[:15]:
        bar = "█" * min(c, 40)
        print(f"    {n:>2} unit(s): {c:>4} sessions  {bar}")

    print()
    print("  保司分布 (实质性 unit, top 20):")
    comp_dist = stats["company_distribution"]
    for company, c in comp_dist.most_common(20):
        print(f"    {company:<16} {c:>4}")

    print()
    print(f"  LLM 待处理: {len(result['llm_pending'])} unit")
    print(f"  处理异常:   {len(result['errors'])} session")
    print("=" * 62)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="P0: Session → Question Units")
    parser.add_argument(
        "--session-id",
        type=str,
        help="处理单个 session(调试用)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="跳过 LLM 上下文推断步骤",
    )
    parser.add_argument(
        "--sessions-dir",
        type=str,
        default=SESSIONS_DIR,
        help="sessions 目录路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="输出目录",
    )
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载
    print(f"加载 sessions 从: {args.sessions_dir}")
    sessions, load_errors = load_sessions(args.sessions_dir)
    print(f"加载了 {len(sessions)} 个 session, {len(load_errors)} 个解析错误")

    if args.session_id:
        # 单 session 调试模式
        target = None
        for session in sessions:
            s = session.get("session", {})
            if s.get("id") == args.session_id:
                target = session
                break
        if target is None:
            print(f"未找到 session: {args.session_id}")
            sys.exit(1)

        units = build_question_units(target)
        print(f"\n切分出 {len(units)} 个 unit:\n")
        for u in units:
            print(f"  [{u['unit_id']}]")
            print(f"    question_raw:      {u['question_raw'][:120]}")
            print(f"    is_substantive:    {u['is_substantive']}")
            print(f"    context_company:   {u['context_company']}")
            print(f"    context_product:   {u['context_product']}")
            print(f"    is_context_inferred: {u['is_context_inferred']}")
            if u.get('inference_basis'):
                print(f"    inference_basis:   {u['inference_basis']}")
            print()

        # 也输出完整 JSON 到文件
        out_path = os.path.join(args.output_dir, "p0_debug_single.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(units, f, ensure_ascii=False, indent=2)
        print(f"完整输出: {out_path}")
        return

    # 全量处理
    print("开始切分...")
    result = process_all_sessions(sessions)

    # 打印汇总
    print_summary(result)

    # 写入 question_units.json
    units_path = os.path.join(args.output_dir, "question_units.json")
    with open(units_path, "w", encoding="utf-8") as f:
        json.dump(result["units"], f, ensure_ascii=False, indent=2)
    print(f"\n写入 question units: {units_path} ({len(result['units'])} 条)")

    # 写入 summary.json
    summary_path = os.path.join(args.output_dir, "p0_summary.json")
    summary_data = {
        "generated_at": datetime.now().isoformat(),
        "sessions_dir": args.sessions_dir,
        "stats": dict(result["stats"]),
        "company_distribution": dict(result["stats"]["company_distribution"]),
        "units_per_session_distribution": dict(
            result["stats"]["units_per_session_distribution"]
        ),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"写入汇总统计:      {summary_path}")

    # 写入 llm_pending.json(如有)
    if result["llm_pending"]:
        llm_path = os.path.join(args.output_dir, "p0_llm_pending.json")
        with open(llm_path, "w", encoding="utf-8") as f:
            json.dump(result["llm_pending"], f, ensure_ascii=False, indent=2)
        print(f"写入 LLM 待处理:   {llm_path} ({len(result['llm_pending'])} 条)")
    else:
        print("无需 LLM 上下文推断(规则已覆盖全部 unit)")

    # 写入 errors.json(如有)
    if result["errors"]:
        err_path = os.path.join(args.output_dir, "p0_errors.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(result["errors"], f, ensure_ascii=False, indent=2)
        print(f"写入处理异常:      {err_path} ({len(result['errors'])} 条)")

    print("\nDone.")


if __name__ == "__main__":
    main()
