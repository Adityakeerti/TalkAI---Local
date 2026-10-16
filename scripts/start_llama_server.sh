#!/usr/bin/env bash
# TalkAI - Launch local llama-server with Gemma-4-E2B and CUDA GPU offloading

LLAMA_SERVER="/home/aditya/llama-fork/build/bin/llama-server"
MODEL="/home/aditya/Desktop/STUDY/TalkAI/Gemma-4-E2B/gemma-4-E2B_q4_0-it.gguf"
MMPROJ="/home/aditya/Desktop/STUDY/TalkAI/Gemma-4-E2B/gemma-4-E2B-it-mmproj.gguf"

if [ ! -f "$LLAMA_SERVER" ]; then
    echo "Error: llama-server binary not found at $LLAMA_SERVER"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found at $MODEL"
    exit 1
fi

echo "========================================================"
echo "Starting llama-server for Gemma-4-E2B (Apache 2.0)"
echo "Binary:  $LLAMA_SERVER"
echo "Model:   $MODEL"
echo "MMProj:  $MMPROJ"
MERGED_MODEL="/home/aditya/Desktop/STUDY/TalkAI/models/duplex_gemma_merged.gguf"
if [ -f "$MERGED_MODEL" ]; then
    echo "Found merged model: $MERGED_MODEL"
    MODEL="$MERGED_MODEL"
    LORA_ARGS=""
else
    LORA="/home/aditya/Desktop/STUDY/TalkAI/models/duplex_gemma_lora.gguf"
    LORA_ARGS=""
    if [ -f "$LORA" ]; then
        echo "LoRA:    $LORA"
        LORA_ARGS="--lora $LORA"
    fi
fi

echo "Port:    8080 (http://127.0.0.1:8080)"
echo "GPU offload: Full layers (-ngl 99)"
echo "Slots / Ctx: 1 slot (-np 1), 2048 ctx (VRAM <= 3.8 GB)"
echo "========================================================"

exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  $LORA_ARGS \
  -ngl 99 \
  --ctx-size 2048 \
  -np 1 \
  --host 0.0.0.0 \
  --port 8080
