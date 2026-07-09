# 运行环境与依赖版本

- memory_id: `mem_g_env`
- created_or_updated_at: `2026-06-05T09:00:00+00:00`

## Insight

项目统一使用 conda 环境 agent：Python 3.10、torch 2.7.1+cu118、transformers 5.12、numpy 2.2、PyYAML 6.0。不安装 faiss 和 sentence-transformers，离线环境下模型local_files_only 加载。新同学入组先 conda activate agent 再跑 requirements.txt 核对版本。
