# 审计签署记录

- **复核人**：王饮冰（论文第一作者）
- **日期**：2026-10-23
- **范围**：42 条自由对话 episode
  - 主包：E01–E30（28 条成交、2 条未成交）
  - 负例补审：N01–N12（12 条未成交）
- **复核方式**：通过 `review_zh.html` 逐条查看中文复述与英文原文，对照“待复核判定”
- **结果**：**42/42 认可，无需修改**
- **判定标准**：同一轮内，双方是否明确接受同一个分割

## 复核人导出的原始结果

```
E01,ok
E02,ok
E03,ok
E04,ok
E05,ok
E06,ok
E07,ok
E08,ok
E09,ok
E10,ok
E11,ok
E12,ok
E13,ok
E14,ok
E15,ok
E16,ok
E17,ok
E18,ok
E19,ok
E20,ok
E21,ok
E22,ok
E23,ok
E24,ok
E25,ok
E26,ok
E27,ok
E28,ok
E29,ok
E30,ok
N01,ok
N02,ok
N03,ok
N04,ok
N05,ok
N06,ok
N07,ok
N08,ok
N09,ok
N10,ok
N11,ok
N12,ok
```

## 相关材料

- 中文复核页：`review_zh.html`
- 盲评材料：`audit_pack.txt`、`audit_pack_neg.txt`
- 标签：`labels.csv`、`labels_neg.csv`
- 裁决器输出：`judge.csv`
- 中文复述：`summaries_zh.json`

## 说明（写论文时的口径）

- 42 条标签的制作过程**对任何自动判定（旧解析器、裁决器）都是盲的**。
- 复核人这次的确认**不是盲审**——复核时看得到标签，因此它能排除明显错误，但不能排除细微偏差。论文 Limitations 已如实写明这一点。
