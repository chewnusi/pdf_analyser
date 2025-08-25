#!/bin/bash
set -e

if [ -n "$HF_TOKEN" ] && [ ! -f /root/.cache/huggingface/token ]; then
    echo "Logging in to HuggingFace at runtime..."
    huggingface-cli login --token $HF_TOKEN
    echo "HuggingFace login successful"
elif [ ! -f /root/.cache/huggingface/token ]; then
    echo "Warning: No HuggingFace token found. Some models may not be downloadable."
else
    echo "HuggingFace token already configured"
fi

exec streamlit run main.py
