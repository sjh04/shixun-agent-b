# Conversation meta_command

- memory_id: `mem_conversation_meta_command`
- conversation_id: `meta_command`
- created_or_updated_at: `2026-07-13T04:09:54+02:00`
- flagged: `false`

## Change Report

```json
{
  "change_type": "new",
  "duplicate": false,
  "conflict": false,
  "similarity": null,
  "method": "rule",
  "notes": []
}
```

## Final Answer

已确认完整检索评测需要使用专用环境运行消融脚本，并只选择完整配置行。

## Messages

```json
[
  {
    "role": "user",
    "content": "怎么跑 Qwen 检索评测？"
  }
]
```

## Trace

```json
{
  "tool_names": [
    "exec_command"
  ],
  "command": "/home/ubuntu/miniconda3/envs/agent/bin/python run_b5_ablation.py --llm on --only FULL"
}
```
