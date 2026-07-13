# B5 整合层 / Poison Gate 评测（judge=off）

## RQ3 三分类（change_report）

- 样例数：18
- 准确率：0.389
- 冲突检出率：0.167

混淆矩阵（行=标注，列=预测）：

| 标注\预测 | duplicate | supplement | conflict |
| --- | --- | --- | --- |
| duplicate | 0 | 5 | 1 |
| supplement | 0 | 6 | 0 |
| conflict | 0 | 5 | 1 |

## RQ4 Poison Gate

- 投毒样例：10，正常样例：10
- 拦截率 TPR：0.000
- 误杀率 FPR：0.400
