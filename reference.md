# 港险条款问答 RAG 评测基准集 — 字段规范（Field Spec）

> 版本：v1.0（已用生产环境真实数据校准）
> 知识库版本基准：`rag_v1_260604_bge_m3`
> 数据来源：企业微信真实咨询（org_code = AWM / awm）

---

## 0. 这份规范怎么来的（给团队和老师的背景）

本字段集不是凭空设计，而是经过四轮校准定稿：

1. 从「通用 QA 测评集」出发，确立检索层 + 生成层双层评测框架；
2. 用企微真实对话样本（2052 个 session）校准「样本形态」——发现样本分五类（正常/拒答/综合/闲聊/系统报错），必须先分流；
3. 吸收阿里天池 AFAC2024「保险条款问答」赛题的评分思路——答案拆成「核心事实 + 关键词 + 语义相似」三层；
4. **连入生产向量库（DashVector）、OSS、业务中台实测**——把所有「靠猜」的字段换成实测确认的真实键名。

**实测确认的关键事实（字段设计的依据）：**

| 事实 | 实测证据 | 对字段的影响 |
|---|---|---|
| `document_id` 在库里恒为空 | 所有召回结果 `document_id=""` | 主键改用 `hash` |
| `hash` 是文件级主键，贯穿向量库+OSS | 每条都有 hash，`json_oss_path=hash_jsons/{hash}.json` | `gold_evidence` 用 `hash` 定位 |
| `doc.id`（uuid）是 chunk 级标识，库重建即失效 | 同 hash 不同页 → 不同 doc.id | 不用 doc.id 标 gold，改用稳定的 `(hash, page)` |
| `page` 是 1-based | metadata.page=10 对应 OSS page_index=9 | `gold_evidence.page` 口径=1-based |
| 同一文件多页独立召回 | 同 hash 的 p4/p10/p11 各为独立结果 | recall 粒度 = `(hash, page)` |
| `type` 编码版本月份 | 优惠类文档 type=`2026-05`/`2026-06` 等 | 新增 `doc_type` 字段锁版本 |
| 用户口语 ≠ 条款术语 | 「预存利率」库中正式名为「保证利息」 | 新增 `needs_term_mapping` 标记 |

---

## 1. 字段总览（六层）

| 层 | 作用 | 字段数 |
|---|---|---|
| A 样本管理层 | 闸门：决定样本能不能用、归哪类、怎么评 | 10 |
| B 输入层 | 真实问题原样保留 + 脏特征标注 | 4(+7子) |
| C 检索层 | 检索 ground truth，评 recall/排序/拒答路由 | 6 |
| D 生成层 | 回答 ground truth，三层评分 | 9 |
| E 评分配置 | 不同题型套不同打分公式 | 2 |
| F 标注元信息 | 标注溯源与质量 | 3 |

---

## 2. 逐字段说明

### A. 样本管理层

| 字段 | 类型 | 取值/示例 | 说明 |
|---|---|---|---|
| `id` | string | `OBJ-0118` | 评测集内唯一 ID |
| `session_id` | string | `144263597516689518` | 复用企微 session id，可回溯原始对话 |
| `sample_type` | enum | `normal_answer` / `correct_refusal` / `multi_doc` / `chitchat` / `system_error` | **第一闸门**：先分流再标注 |
| `is_evaluable` | bool | `true` / `false` | `system_error`/空session/闲聊 → false，不计入主指标 |
| `source_company` | string | `宏利保险` | 保司标准中文名（对齐中台 19 家值域 / metadata.company） |
| `source_company_id` | int | `2` | 链接中台标准保司 id（中台 API 可用时填） |
| `product_name` | string | `宏挚传承保障计划` | 产品标准名（对齐中台 770 产品值域） |
| `product_id` | int | `2910` | 链接中台标准产品 id（可空，中台不稳时） |
| `kb_version` | string | `rag_v1_260604_bge_m3` | **关键**：标注时所用向量库版本，防跨版本错位 |
| `org_code` | string | `awm`（请求侧）/ `AWM`（存储侧） | 注意大小写映射：请求用小写，metadata 存大写 |

> **`sample_type` 判定优先级**：先看是否系统报错（含 traceback/error）→ `system_error`；再看是否纯寒暄/无效（"你好""已有记录"）→ `chitchat`；再看是否跨多保司综合（"哪些保司支持…"）→ `multi_doc`；再看是否应拒答 → `correct_refusal`；其余 → `normal_answer`。

### B. 输入层

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `question_raw` | string | `宏利宏挚传承2年交的预存利率系几多` | **铁律：用户原话，一字不改**（含错别字/口语/粤语）。评测喂给模型的就是它 |
| `question_lang` | string | `zh-Hans` / `zh-Hant` / `粤语夹杂` | 提问语言 |
| `question_type` | string | `单业务细节咨询` | 复用 session.question_types |
| `question_sub_type` | string | — | 复用 sub_types（部分 session 才有） |
| `question_quality` | object | 见下 | 真实问题的「脏」特征标注 |

**`question_quality` 子字段：**

| 子字段 | 类型 | 说明 |
|---|---|---|
| `is_underspecified` | bool | 信息不全（没说哪个计划/年期） |
| `has_false_premise` | bool | 预设错误前提（"agent说没等待期"） |
| `is_multi_intent` | bool | 一句多问 |
| `is_cross_company` | bool | 跨保司题 |
| `is_colloquial` | bool | 口语/粤语 |
| `has_typo` | bool | 错别字/谐音 |
| `needs_term_mapping` | bool | **新增**：需口语→条款术语映射（"预存利率"→"保证利息"）。RAG 最易失败处，单独统计 |

### C. 检索层 ground truth

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `gold_evidence` | array | 见下 | **人工核验后**的标准证据，非系统输出 |
| `requires_multi_doc` | bool | `false` | 跨文档综合题（同性配偶案例=true） |
| `no_answer_in_corpus` | bool | `false` | 语料确实没有 → 正确行为是拒答。**必须人工去库里查证，不能信系统的 has_failed_retrieval** |
| `retrieval_granularity` | string | `hash_page` | recall 计算粒度（固定 = (hash,page)） |
| `retrieval_candidates_snapshot` | array | — | 系统当时 topk 原样存档，**仅历史参考，不当 ground truth**（企微旧库与当前库错位） |
| `kb_version_for_evidence` | string | `rag_v1_260604_bge_m3` | 本条 gold 在哪个库版本上标的 |

**`gold_evidence[]` 子字段：**

| 子字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `hash` | string | `b8062eeb79116a07011af54b8a63933dbae1cc6b` | **文件级主键**，跨库稳定，反查 OSS |
| `page` | int | `1` | **1-based**（已实测） |
| `doc_type` | string | `2026-05` | 即 metadata.type，版本/月份标识，**港险多版本必标** |
| `node_path` | string | `宏利保险 / 优惠信息 / 2026-05` | 目录路径，含保司/类型/产品四级 |
| `file_name` | string | `宏利-宏挚...优惠至5.10.pdf` | 冗余，给标注员看 |

### D. 生成层 ground truth

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `expected_behavior` | enum | `answer` / `refuse` / `clarify` / `correct_premise` | 期望行为 |
| `base_answer` | string | `第2个保单年度10%（2年缴）` | **核心事实，错则一票否决判0分**（吸收赛题思路）。港险需限定版本/年期 |
| `required_keywords` | array | `["宏挚传承","2年缴","保证利息"]` | 关键实体，命中加分 |
| `acceptable_answers` | array | `["...","..."]` | 多个等价表述，算语义相似度取上限 |
| `key_points` | array | `["...","..."]` | 复杂/综合题按要点逐条给分（multi_doc 用此，非 base/kw） |
| `forbidden_content` | array | `["全额赔付","无等待期"]` | 不该出现的幻觉点（拒答题尤其） |
| `currency` | string | `USD` | 港险关键：美元保单常见 |
| `currency_required` | bool | `true` | 涉及保额/价值时，答案是否必须带币种 |
| `clarify_question` | string | — | expected_behavior=clarify 时的理想反问 |

### E. 评分配置

| 字段 | 类型 | 取值 | 说明 |
|---|---|---|---|
| `eval_method` | enum | `objective_qa` / `refusal` / `clarify` / `multi_point` | 题型对应的打分逻辑 |
| `score_weights` | object | `{base:0.25, keyword:0.25, similarity:0.5}` | objective_qa 权重（默认沿用赛题配置） |

> **eval_method 与 sample_type 的对应**：
> - `normal_answer` + 客观题 → `objective_qa`（base+keyword+相似度）
> - `correct_refusal` → `refusal`（判是否正确拒答+无幻觉，base_answer 留空）
> - `multi_doc` → `multi_point`（key_points 覆盖率）
> - clarify 类 → `clarify`（判是否正确反问）

### F. 标注元信息

| 字段 | 类型 | 说明 |
|---|---|---|
| `annotator` | string | 标注人 |
| `verified` | bool | 是否双人核验通过；未核验不进正式评测 |
| `notes` | string | 出题意图、答案解释、边界判断记录 |

---

## 3. 真实样本（经典案例：术语对齐 + 拒答翻转）

> 这是本评测集最有代表性的一条。它来自企微真实对话，系统当时回答"找不到预存利率信息"。
> 但连入生产库实测发现：库里资料充分（召回 score 0.78，排第一），「预存利率」的条款正式名是「保证利息」。
> **系统当时的"拒答"实为术语未对齐导致的漏答 → 样本翻转标为 normal_answer。**

```jsonc
{
  "id": "TERM-0007",
  "session_id": "144263597516689518",
  "sample_type": "normal_answer",          // ★翻转:非correct_refusal
  "is_evaluable": true,
  "source_company": "宏利保险",
  "product_name": "宏挚传承保障计划",
  "kb_version": "rag_v1_260604_bge_m3",
  "org_code": "awm",

  "question_raw": "宏利宏挚传承2年交的预存利率是多少",
  "question_lang": "zh-Hans",
  "question_type": "单业务细节咨询",
  "question_quality": {
    "needs_term_mapping": true,            // ★"预存利率"→"保证利息"
    "is_underspecified": false
  },

  "gold_evidence": [
    {
      "hash": "b8062eeb79116a07011af54b8a63933dbae1cc6b",
      "page": 1,
      "doc_type": "2026-05",               // ★版本锁定:2026年5月优惠版
      "node_path": "宏利保险 / 优惠信息 / 2026-05",
      "file_name": "宏利-宏挚&宏浚預繳保費保證優惠息率推廣優惠（2,3年期）-优惠至5.10.pdf"
    }
  ],
  "requires_multi_doc": false,
  "no_answer_in_corpus": false,            // ★人工核验:库里确实有
  "retrieval_granularity": "hash_page",
  "kb_version_for_evidence": "rag_v1_260604_bge_m3",

  "expected_behavior": "answer",
  "base_answer": "宏挚传承2年缴付期，第2个保单年度保证利息10%（指定计划可额外+1%），保单货币为美元",
  "required_keywords": ["宏挚传承", "2年缴", "保证利息"],
  "acceptable_answers": [
    "宏挚传承2年交，第2个保单年度可享10%保证利息，符合指定计划再加1%。",
    "2年缴付期的宏挚传承，预缴保费第2保单年度保证利息为10%。"
  ],
  "forbidden_content": ["找不到", "无相关信息", "6%", "6.5%"],  // 6%/6.5%是3年缴的数字,2年缴答这个即错
  "currency": "USD",
  "currency_required": true,

  "eval_method": "objective_qa",
  "score_weights": { "base": 0.25, "keyword": 0.25, "similarity": 0.5 },

  "annotator": "TBD",
  "verified": false,
  "notes": "经典术语对齐案例。考察点:(1)能否将口语'预存利率'映射到条款术语'保证利息';(2)能否区分2年缴(10%)与3年缴(6%~6.5%)的不同数值;(3)答案须带版本(2026-05)与币种(USD)。系统当时拒答=漏答,非正确行为。"
}
```

---

## 4. 落地注意事项

1. **先分流再标注**：2052 个 session 不要一次全标。先批量打 `sample_type`，隔离 `system_error` 与空 session，从每类抽 20~30 条做标注试点，磨出判定指引并算双人一致性（Kappa），再铺开。

2. **库版本错位是头号风险**：企微数据导出于 5/25，当前库为 6/4，且库在持续重建。每次重建后抽查 gold_evidence 的 hash 是否仍存在，失效的重标。`kb_version` 字段是评测集长期可用的保命字段。

3. **中台 API 不稳定**：实测出现 SSL 中断，时通时断。`source_company`/`product_name` 优先用 DashVector metadata 里的现成字段，中台仅作补充，不作硬依赖。

4. **库内容构成需确认**：两次抽样召回的 `kind` 全是「产品」「优惠信息」等营销物料，未见「保单契约」条款全文。建议按 `kind`/`category` 拉全库分布，确认「条款类」文档覆盖度——这决定评测集里"除外责任/赔付条件"类题能占多大比例，也决定多少题会合法地 `no_answer_in_corpus`。

5. **术语对齐是独立维度**：「预存利率=保证利息」非个例（等待期/等候期、现金价值/退保价值、受保人/被保人均如此）。`needs_term_mapping` 标记的题应单独统计准确率，这是 RAG 检索最易失败、最该优化的环节。

6. **page 对齐**：反查 OSS 整页原文时，`OSS page_index = metadata.page - 1`（paddle_pages_array 格式）。