# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fish Speech is a multilingual text-to-speech (TTS) system using a Dual-Autoregressive (Dual-AR) architecture:
- **Slow AR (LLaMA-based)**: 4B parameter transformer that predicts semantic tokens along time axis
- **Fast AR**: 400M parameter model that generates residual codebooks for acoustic details
- **Audio Codec**: Descript Audio Codec (DAC) with 10 codebooks at ~21 Hz for encoding/decoding audio

## Development Setup

### Installation

```bash
# System dependencies
apt install portaudio19-dev libsox-dev ffmpeg

# Conda environment (recommended)
conda create -n fish-speech python=3.12
conda activate fish-speech
pip install -e .[cu129]  # or [cu126], [cu128], [cpu]

# Or using UV (faster)
uv sync --python 3.12 --extra cu129
```

### Code Quality

Pre-commit hooks run automatically on commits:
```bash
pre-commit install
pre-commit run --all-files  # manual run
```

Tools: isort (profile=black), black, standard pre-commit hooks

## Architecture

### Core Components

1. **`fish_speech/models/text2semantic/`** - Text to semantic tokens
   - `llama.py`: BaseTransformer model (Slow AR, 4B params)
   - `lit_module.py`: LightningModule wrapper for training
   - `inference.py`: Inference entry point for semantic token generation
   - `lora.py`: LoRA configuration for fine-tuning

2. **`fish_speech/models/dac/`** - Audio codec
   - `modded_dac.py`: DAC encoder/decoder model
   - `inference.py`: VQ encoding/decoding, reconstruction from semantic tokens
   - `rvq.py`: Residual Vector Quantization utilities

3. **`fish_speech/conversation.py` & `content_sequence.py`** - Multi-turn conversation handling
   - `Message`: Role-based messages (system/user/assistant) with mixed text/audio content
   - `Conversation`: Manages multi-turn conversations with proper token encoding
   - `ContentSequence`: Handles text, audio, VQ parts with loss masking

4. **`fish_speech/datasets/`** - Dataset handling
   - `semantic.py`: Semantic dataset classes, protobuf data loading
   - `vqgan.py`: VQGAN-specific datasets

5. **`tools/server/`** - API server (Kui framework)
   - `api_server.py`: Main HTTP server entry point
   - `model_manager.py`: Model loading and lifecycle management
   - `views.py`: API route definitions (/v1/tts, /v1/vqgan/encode, /v1/vqgan/decode)

6. **`tools/llama/`** - LLaMA-specific tools
   - `build_dataset.py`: Pack audio/text into protobuf format for training
   - `merge_lora.py`: Convert LoRA weights to regular checkpoints

### Configuration System

Hydra-based configs in `fish_speech/configs/`:
- `base.yaml`: Base training configuration (trainer, callbacks, logger defaults)
- `text2semantic_finetune.yaml`: Fine-tuning config (uses LoRA)
- `lora/r_8_alpha_16.yaml`: LoRA rank/alpha settings

Key patterns:
- `_target_`: specifies class to instantiate
- `_partial_: true`: creates partial function for delayed instantiation
- Config composition via `defaults:` list

## Common Commands

### Inference

```bash
# Download model weights
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro

# Command line inference (3 steps)
# 1. Encode reference audio to VQ tokens
python fish_speech/models/dac/inference.py -i "test.wav" --checkpoint-path "checkpoints/s2-pro/codec.pth"

# 2. Generate semantic tokens from text
python fish_speech/models/text2semantic/inference.py \
    --text "Your text here" \
    --prompt-text "Reference text" \
    --prompt-tokens "fake.npy" \
    --compile  # Optional: faster inference

# 3. Decode semantic tokens to audio
python fish_speech/models/dac/inference.py -i "codes_0.npy"

# WebUI (Gradio)
python tools/run_webui.py --compile

# API Server
python tools/api_server.py \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --listen 0.0.0.0:8080
```

### Training/Fine-tuning

```bash
# 1. Extract semantic tokens from dataset
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"

# 2. Pack dataset into protobuf
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16

# 3. Fine-tune with LoRA
python fish_speech/train.py --config-name text2semantic_finetune \
    project=my_project \
    +lora@model.model.lora_config=r_8_alpha_16

# 4. Merge LoRA weights for inference
python tools/llama/merge_lora.py \
    --lora-config r_8_alpha_16 \
    --base-weight checkpoints/s2-pro \
    --lora-weight results/my_project/checkpoints/step_000000010.ckpt \
    --output checkpoints/s2-pro-finetuned/
```

### Docker

```bash
# WebUI
docker compose --profile webui up

# API server
docker compose --profile server up

# With compile optimization
COMPILE=1 docker compose --profile server up
```

## Key Patterns

### Multi-speaker and Multi-turn
- Speaker tokens: `<|speaker:i|>` where `i` is speaker ID
- Conversation format: `<|im_start|>role\nmodality<|im_end|>` wrapping
- Interactive mode controlled by `interactive_prob` in dataset config

### Emotion/Prosody Tags
Inline control via `[tag]` syntax: `[whisper]`, `[excited]`, `[laughing]`, etc.
Tags are embedded directly in text input.

### Loss Calculation
- `cal_loss` attribute controls which tokens contribute to loss
- Special tokens (im_start, modality) typically excluded from loss
- Multi-codebook loss handling for RVQ training

## Model Weights Structure

```
checkpoints/s2-pro/
├── codec.pth          # DAC encoder/decoder weights
├── tokenizer.tiktoken # Tokenizer model
├── model.pth          # LLaMA Slow AR weights (base)
└── ...
```

## Important Notes

- **Do NOT fine-tune RL-trained models** - shifts distribution, degrades performance
- **Minimum 24GB GPU VRAM** for inference with full model
- **Torch.compile** provides ~10x speedup but not supported on Windows/macOS
- **Windows users**: Use `trainer.strategy.process_group_backend=gloo` for training
- **For GPUs without bf16**: Use `--half` flag for fp16 mode
