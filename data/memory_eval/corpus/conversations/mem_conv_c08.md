# ModuleNotFoundError yaml 解决办法

- memory_id: `mem_conv_c08`
- conversation_id: `eval_c08`
- created_or_updated_at: `2026-06-11T10:00:00+00:00`

## Final Answer

报 ModuleNotFoundError: No module named 'yaml' 有两个常见原因：一是包名装错，正确的安装命令是 pip install pyyaml，包名是 pyyaml 而导入名是 yaml，直接 pip install yaml 装到的是另一个废弃包；二是忘了激活 conda 环境，在 base 里跑了脚本。先 conda activate agent 再确认 pip list 里有 PyYAML。
