#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF="${SCRIPT_DIR}/.venv/bin/hf"

if [[ ! -x "$HF" ]]; then
  echo "hf が見つかりません: $HF"
  echo "先にプロジェクトの venv を用意してください (python3 -m venv .venv && .venv/bin/pip install huggingface_hub)"
  exit 1
fi

DEST=/home/sarulab/kengo_takemoto/data/SAKURA
mkdir -p "$DEST"

for repo in AnimalQA EmotionQA GenderQA LanguageQA; do
  echo "Downloading SLLM-multi-hop/${repo} ..."
  "$HF" download "SLLM-multi-hop/${repo}" \
    --repo-type dataset \
    --local-dir "${DEST}/${repo}"
done
