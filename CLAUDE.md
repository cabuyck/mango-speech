# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fish Speech is a state-of-the-art multilingual text-to-speech (TTS) system using a **Dual-Autoregressive (Dual-AR) architecture**:

- **Slow AR (LLAMA transformer, ~4B params)**: Operates along the time axis, predicting the primary semantic codebook
- **Fast AR (DAC decoder, ~400M params)**: Generates the remaining 9 residual codebooks at each time step using RVQ (Residual Vector Quantization)

The model supports 80+ languages, sub-word level prosody control via natural language tags (e.g., `[whisper]`, `[excited]`), multi-speaker generation, and multi-turn conversations.

### RPC-Based Layer Offload

Fish Speech supports **RPC-based layer offload** for distributing the Slow AR transformer across two machines:
- **Host**: Embeddings + first N transformer layers
- **VM Worker**: Remaining layers + norm + output + Fast AR

See `docs/RPC_OFFLOAD.md` for detailed documentation on using RPC mode.

## Installation and Setup

```bash
# Install dependencies using uv (recommended)
uv pip install -e .

# For specific CUDA versions
uv pip install -e ".[cu126]"  # CUDA 12.6
uv pip install -e ".[cu128]"  # CUDA 12.8
uv pip install -e ".[cu129]"  # CUDA 12.9
uv pip install -e ".[cpu]"    # CPU-only
```

## Development Commands

### Running the API Server

```bash
python tools/api_server.py \
    --llama-checkpoint-path checkpoints/fish-speech-2.0 \
    --decoder-checkpoint-path checkpoints/fish-speech-2.0/firefly-gan-vq-fsq-4x1024-42hz-generator.pth \
    --decoder-config-name firefly_gan_vq \
    --device cuda \
    --half
```

### Running the WebUI

```bash
# Build the frontend first
cd awesome_webui
npm install
npm run build
cd ..

# Then run the WebUI
python tools/run_webui.py \
    --llama-checkpoint-path checkpoints/fish-speech-2.0 \
    --decoder-checkpoint-path checkpoints/fish-speech-2.0/firefly-gan-vq-fsq-4x1024-42hz-generator.pth \
    --decoder-config-name firefly_gan_vq
```

### Training

```bash
# Train text2semantic model (LLAMA)
python -m fish_speech.train --config-name text2semantic_finetune

# Train VQGAN/decoder
python -m fish_speech.train --config-name modded_dac_vq
```

### Linting and Formatting

```bash
# Run pre-commit hooks
pre-commit run --all-files

# Install pre-commit hooks
pre-commit install
```

The project uses:
- `isort` for import sorting (with black profile)
- `black` for code formatting
- `mixed-line-ending` to enforce LF line endings

## Architecture

### Core Components

**`fish_speech/models/text2semantic/`** - The LLAMA-based text-to-semantic model
- `llama.py`: BaseTransformer class with LLaMA architecture
- `inference.py`: Inference logic for generating semantic tokens
- `lit_module.py`: LightningModule wrapper for training

**`fish_speech/models/dac/`** - The audio decoder (modified Descript Audio Codec)
- `modded_dac.py`: DAC encoder/decoder with RVQ support
- `rvq.py`: Residual Vector Quantization implementation
- `inference.py`: Loading and running the decoder model

**`fish_speech/inference_engine/`** - TTS inference orchestration
- `__init__.py`: TTSInferenceEngine combines LLAMA + DAC for full TTS pipeline
- `reference_loader.py`: Handles reference audio loading and caching
- `vq_manager.py`: VQ token decoding utilities

**`tools/server/`** - API server implementation
- `model_manager.py`: ModelManager loads and manages both models
- `views.py`: HTTP endpoints (`/v1/health`, `/v1/vqgan/encode`, `/v1/tts`)
- `inference.py`: Inference wrapper for API requests

### Data Flow

1. **Text → Semantic Tokens**: LLAMA model takes text + reference audio VQ codes → generates semantic tokens
2. **Semantic Tokens → Audio**: DAC decoder takes semantic tokens → reconstructs audio waveform

### Configuration System

Uses **Hydra** for configuration management:
- Base config: `fish_speech/configs/base.yaml`
- Text2semantic finetune: `fish_speech/configs/text2semantic_finetune.yaml`
- DAC VQ training: `fish_speech/configs/modded_dac_vq.yaml`
- LoRA configs: `fish_speech/configs/lora/`

Config values support `eval` resolver and can reference other configs via `${path.to.value}`.

### Training with PyTorch Lightning

- Entry point: `fish_speech/train.py`
- Uses Hydra for config composition
- Supports DDP multi-GPU training
- Checkpoints saved to `results/{project}/checkpoints/`
- Auto-resumes from latest checkpoint if available

### Conversation and Content Encoding

**`fish_speech/conversation.py`** - Message-based conversation API
- Message types: `system`, `user`, `assistant`
- Supports multi-modal content (text, audio, VQ codes)
- Handles interleaved text/audio conversations

**`fish_speech/content_sequence.py`** - Low-level content encoding
- Part types: `TextPart`, `VQPart`, `AudioPart`
- Encodes conversations to token sequences with loss masks

### Tokenizer

**`fish_speech/tokenizer.py`** - FishTokenizer wraps tiktoken
- Uses custom tokenizer from checkpoint directory
- Special tokens: `<|im_start|>`, `<|im_end|>`, modality tokens

## Docker Development

```bash
# Build dev image
docker build -f dockerfile.dev -t fish-speech:dev .

# Use Docker Compose
docker compose -f compose.yml up
```

## Key Files to Understand

- `fish_speech/models/text2semantic/llama.py` - Core LLAMA model architecture
- `fish_speech/models/dac/modded_dac.py` - Audio codec architecture
- `fish_speech/inference_engine/__init__.py` - How the two models combine for inference
- `tools/server/views.py` - API endpoint implementations
- `fish_speech/conversation.py` - Conversation/message handling API
- `fish_speech/configs/text2semantic_finetune.yaml` - Example training config

## Important Notes

- Models are loaded from HuggingFace format checkpoints (with `config.json` and model weights)
- The `Dual-AR` architecture means the LLAMA model generates only the first codebook, then the DAC generates the remaining 9 codebooks
- Reference audio is encoded to VQ codes and used as conditioning for the LLAMA model
- Streaming inference is supported via the `streaming` parameter in ServeTTSRequest
- Thread-safe LLAMA inference uses a queue pattern (see `launch_thread_safe_queue`)
