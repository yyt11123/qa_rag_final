# P5 答案标注指引

> 生成时间: 2026-06-22T19:13:26.075370
> 待标注样本数: 206

---

## 标注总览

| 指标 | 值 |
|---|---|
| 总样本数 | 206 |
| objective_qa 题 | 100 |
| refusal 题 | 81 |
| multi_point 题 | 25 |
| clarify 题 | 0 |

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
