#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARCHIVE="${REPO_ROOT}/demo_checkpoints/acceptance_demo_gpu_20260713.tar.gz"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "checkpoint archive not found: ${ARCHIVE}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
tar -xzf "${ARCHIVE}"

echo "Restored acceptance demo checkpoint:"
echo "  ${ARCHIVE}"
echo
echo "Key docs:"
echo "  组长验收演示方案.md"
echo "  B5验收演示方案.md"
