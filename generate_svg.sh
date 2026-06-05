#!/bin/bash
# Convenient wrapper to generate SVGs using Qwen2.5-Coder

if [ -z "$1" ]; then
    echo "Usage: ./generate_svg.sh <prompt_subject> [output_file.svg] [model_id]"
    echo "Example: ./generate_svg.sh bird output.svg"
    exit 1
fi

SUBJECT="$1"
OUTPUT="${2:-output.svg}"
MODEL="${3:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"

PYTHON_BIN="/media/jang/home/Deve/torch/bin/python"
SCRIPT_PATH="/media/jang/home/Deve/zen-lights/internal/paint/svg/generate_svg.py"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python binary not found at $PYTHON_BIN"
    exit 1
fi

echo "Running SVG generation for subject: '$SUBJECT'..."
"$PYTHON_BIN" "$SCRIPT_PATH" "$SUBJECT" "$OUTPUT" "$MODEL"
