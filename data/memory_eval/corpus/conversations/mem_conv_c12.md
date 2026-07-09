# BM25 参数讨论

- memory_id: `mem_conv_c12`
- conversation_id: `eval_c12`
- created_or_updated_at: `2026-06-17T10:00:00+00:00`

## Final Answer

B5 检索层的 BM25 参数定为 k1=1.5、b=0.75，是信息检索文献里的经典默认值。k1 控制词频饱和速度，中文字符 bigram 的词频分布偏平，1.5 够用；b 控制文档长度归一化强度，语料里长短文档混杂，0.75 能压住长文档的天然优势。调参优先级不高，先把 chunk 和融合做好收益更大。
