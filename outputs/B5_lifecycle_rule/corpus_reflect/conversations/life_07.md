# 精度选型

## Final Answer

bfloat16 全程无溢出警告，float16 在长文任务偶发溢出，配置锁定 bfloat16。
