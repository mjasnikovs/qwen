#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

HF_REPO="${HF_REPO:-unsloth/Qwen3-VL-32B-Instruct-GGUF}"
HF_FILE="${HF_FILE:-Qwen3-VL-32B-Instruct-Q4_K_M.gguf}"
HF_MMPROJ="${HF_MMPROJ:-mmproj-F16.gguf}"
MODEL_FILE="${MODEL_FILE:-qwen3-vl-32b.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-qwen3-vl-32b-mmproj.gguf}"

mkdir -p models

auth_header=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  auth_header=(--header "Authorization: Bearer ${HF_TOKEN}")
fi

fetch() {
  local remote="$1" local_name="$2" min_bytes="$3"
  local target="models/${local_name}"

  if [[ -f "$target" ]]; then
    local size
    size=$(stat -c%s "$target")
    if (( size > min_bytes )); then
      echo "Already present: $target ($(numfmt --to=iec --suffix=B "$size"))"
      return 0
    fi
    echo "Existing $target is suspiciously small ($size bytes); re-downloading."
    rm -f "$target"
  fi

  local url="https://huggingface.co/${HF_REPO}/resolve/main/${remote}?download=true"
  echo "Downloading ${remote} -> ${target}"
  wget --progress=bar:force:noscroll \
       "${auth_header[@]}" \
       -c -O "$target" \
       "$url"
  echo "Saved $target"
}

fetch "$HF_FILE" "$MODEL_FILE" $((1024 * 1024 * 1024))   # >1 GB sanity floor

if [[ -n "$HF_MMPROJ" ]]; then
  fetch "$HF_MMPROJ" "$MMPROJ_FILE" $((10 * 1024 * 1024)) # >10 MB sanity floor
else
  echo "HF_MMPROJ is empty — skipping vision projector download."
fi
