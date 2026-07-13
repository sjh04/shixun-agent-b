# B5 整合层 / Poison Gate 评测（judge=on）

## RQ3 三分类（change_report）

- 样例数：18
- 准确率：1.000
- 冲突检出率：1.000

混淆矩阵（行=标注，列=预测）：

| 标注\预测 | duplicate | supplement | conflict |
| --- | --- | --- | --- |
| duplicate | 6 | 0 | 0 |
| supplement | 0 | 6 | 0 |
| conflict | 0 | 0 | 6 |

## RQ4 Poison Gate

- 投毒样例：10，正常样例：10
- 拦截率 TPR：0.900
- 误杀率 FPR：0.100
