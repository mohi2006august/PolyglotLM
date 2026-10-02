#!/usr/bin/env bash
# Convert a merged PolyglotLM model to GGUF and quantise it (FR-07).
#
#   LLAMA_CPP_DIR=~/src/llama.cpp polyglotlm/quant/to_gguf.sh artifacts/indic-3b/models/sft \
#       artifacts/indic-3b/gguf Q4_K_M
#
# The model directory must contain the merged tokenizer files (tokenizer.json): llama.cpp reads
# the vocabulary from them, and a GGUF built from the base tokenizer would not know the new
# tokens. `poly quant` runs the same steps with extra preflight checks.
set -euo pipefail

MODEL_DIR=${1:?usage: to_gguf.sh MODEL_DIR [OUT_DIR] [QUANT]}
OUT_DIR=${2:-gguf}
QUANT=${3:-Q4_K_M}
: "${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to a built llama.cpp checkout}"

for required in config.json tokenizer.json; do
  if [[ ! -f "$MODEL_DIR/$required" ]]; then
    echo "error: $MODEL_DIR/$required is missing (save the merged tokenizer with the model)" >&2
    exit 1
  fi
done

NAME=$(basename "$(dirname "$(dirname "$MODEL_DIR")")")-$(basename "$MODEL_DIR")
mkdir -p "$OUT_DIR"
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" \
  --outfile "$OUT_DIR/$NAME-F16.gguf" --outtype f16
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$OUT_DIR/$NAME-F16.gguf" "$OUT_DIR/$NAME-$QUANT.gguf" "$QUANT"
echo "$OUT_DIR/$NAME-$QUANT.gguf"
