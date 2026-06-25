#!/usr/bin/env bash
# =============================================================================
# vLLM 启动脚本 —— 本地 Qwen3.5-4B (OpenAI 兼容服务)
#
#   说明:
#   - 该模型为多模态 (Qwen3_5ForConditionalGeneration, model_type=qwen3_5),
#     原生上下文 262144,bf16 权重约 9.3GB。
#   - 需要支持 qwen3_5 架构的较新 vLLM (安装: pip install -U vllm)。
#   - 与 agent 工程的 transformers 后端 (configs/model.yaml) 相互独立,
#     仅对外提供 http://HOST:PORT/v1 的 OpenAI 接口。
#
#   用法:
#     bash serve_vllm.sh                 # 用默认参数 (GPU0, 32K 上下文, 8000 端口)
#     GPUS=1 PORT=8001 bash serve_vllm.sh
#     GPUS=0,1 TP=2 MAX_LEN=65536 bash serve_vllm.sh   # 双卡张量并行
# =============================================================================
set -euo pipefail

# ---------------------- 可调参数 (环境变量可覆盖) ----------------------
MODEL_PATH="${MODEL_PATH:-/mnt/aisdata/sjh04/实训2/Qwen3.5-4B}"
SERVED_NAME="${SERVED_NAME:-Qwen3.5-4B}"   # 客户端 model 字段用这个名字
GPUS="${GPUS:-0}"                          # 用哪几张卡: "0" / "0,1" / "2,3"
TP="${TP:-1}"                              # tensor-parallel-size,必须 == GPU 数
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"                # 上下文上限 (原生 262144,按显存收敛)
GPU_UTIL="${GPU_UTIL:-0.90}"              # 单卡显存占用比例
DTYPE="${DTYPE:-bfloat16}"
# ----------------------------------------------------------------------

# 前置检查
if ! command -v vllm >/dev/null 2>&1; then
  echo "[ERR] 未找到 vllm,请先安装: pip install -U vllm" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERR] 模型路径无效 (缺 config.json): ${MODEL_PATH}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "[vLLM] model      = ${MODEL_PATH}"
echo "[vLLM] served-name= ${SERVED_NAME}"
echo "[vLLM] gpus       = ${GPUS}  (tp=${TP})"
echo "[vLLM] max-len    = ${MAX_LEN}   dtype=${DTYPE}   gpu-util=${GPU_UTIL}"
echo "[vLLM] endpoint   = http://${HOST}:${PORT}/v1"
echo

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --dtype "${DTYPE}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --trust-remote-code \
  --chat-template "${MODEL_PATH}/chat_template.jinja" \
  --host "${HOST}" \
  --port "${PORT}"
  # 纯文本 agent 场景可禁用多模态以省显存(新版 vLLM 语法,按需取消注释):
  #   --limit-mm-per-prompt '{"image": 0, "video": 0}'

# -----------------------------------------------------------------------------
# 启动后自测 (另开终端):
#   curl http://127.0.0.1:8000/v1/models
#   curl http://127.0.0.1:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"Qwen3.5-4B","messages":[{"role":"user","content":"你好"}]}'
# -----------------------------------------------------------------------------
