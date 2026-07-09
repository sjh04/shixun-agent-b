# 系统提示词模板位置

- memory_id: `mem_conv_c09`
- conversation_id: `eval_c09`
- created_or_updated_at: `2026-06-09T10:00:00+00:00`

## Final Answer

系统提示词模板放在 prompts/ 目录下，具体用哪个由 runtime_input.json 的 system_prompt_path 字段指定，路径相对输入文件所在目录解析。要换提示词不要直接改默认模板，复制一份改名后把 system_prompt_path 指过去，保证别人的演示不受影响。模板里的占位符由 B1 在注入时填充。
