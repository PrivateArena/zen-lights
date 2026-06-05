#!/bin/bash
export ORT_SHARED_LIB_PATH=/media/jang/home/Deve/zen-tts/piper/libonnxruntime.so.1.24.2

./zen-lights server -addr 127.0.0.1:8765 -config config.json
