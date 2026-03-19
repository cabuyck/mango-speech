# RPC-Based Layer Offload for Fish Speech

This document describes the RPC-based layer offload feature for Fish Speech, which enables distributing the Slow AR (LLAMA) transformer across two machines for inference.

## Overview

Fish Speech uses a **Dual-AR architecture**:
- **Slow AR (LLAMA ~4B params)**: Generates semantic tokens
- **Fast AR (DAC ~400M params)**: Generates residual codebooks

The RPC layer offload feature allows you to split the Slow AR transformer across two workers:
- **Host machine**: Owns embeddings, codebook embeddings, first N transformer layers, and their KV caches
- **VM worker**: Owns remaining layers, their KV caches, norm, output/logits, and the entire fast AR stack

This is useful for scenarios where you want to distribute GPU resources across multiple machines.

## Architecture

```
Host: embeddings → position prep → [layers 0:N-1]
      └─ RPC call (hidden state + metadata) ─┐
                                                ├─→ VM: [layers N:31] → norm → logits → sampling → fast AR
VM: ←─────────────────────────────────────────┘                                          └─→ return token bundle
```

### Data Flow

1. Host processes input tokens through embeddings and first N transformer layers
2. Host sends intermediate hidden states to VM via RPC (one call per decode step)
3. VM completes the slow AR (remaining layers), samples semantic token, and runs fast AR
4. VM returns generated codebooks to host
5. Each worker maintains its own KV caches locally (no cache transfer over RPC)

## Quick Start

### 1. Start the VM Worker

On the VM worker machine, run:

```bash
python tools/launch_rpc_vm.py \
    --checkpoint-path checkpoints/fish-speech-2.0 \
    --device cuda \
    --half \
    --num-host-layers 16 \
    --vm-address localhost:29501 \
    --host-address localhost:29500
```

Replace addresses with actual hostnames/IPs for cross-machine deployment:
```bash
python tools/launch_rpc_vm.py \
    --checkpoint-path checkpoints/fish-speech-2.0 \
    --device cuda \
    --half \
    --num-host-layers 16 \
    --vm-address 192.168.1.100:29501 \
    --host-address 192.168.1.50:29500
```

### 2. Start the Host (API Server)

On the host machine, either use the convenience script:

```bash
./tools/launch_rpc_host.sh
```

Or run the API server directly with RPC flags:

```bash
python tools/api_server.py \
    --mode tts \
    --llama-checkpoint-path checkpoints/fish-speech-2.0 \
    --decoder-checkpoint-path checkpoints/fish-speech-2.0/firefly-gan-vq-fsq-4x1024-42hz-generator.pth \
    --decoder-config-name firefly_gan_vq \
    --device cuda \
    --half \
    --listen 127.0.0.1:8080 \
    --rpc-enable \
    --rpc-role host \
    --rpc-host-address localhost:29500 \
    --rpc-vm-address localhost:29501 \
    --rpc-num-host-layers 16
```

### 3. Make API Requests

The API works the same way as non-RPC mode:

```bash
curl -X POST http://127.0.0.1:8080/v1/tts \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Hello world, this is a test.",
        "references": [],
        "max_new_tokens": 1024
    }'
```

## Configuration Options

### RPC Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--rpc-enable` | flag | False | Enable RPC-based layer offload |
| `--rpc-role` | str | "host" | Role: "host" or "vm" |
| `--rpc-host-address` | str | "localhost:29500" | Host address for RPC (format: host:port) |
| `--rpc-vm-address` | str | "localhost:29501" | VM worker address for RPC (format: host:port) |
| `--rpc-num-host-layers` | int | 16 | Number of transformer layers on host |
| `--rpc-timeout-ms` | int | 60000 | RPC timeout in milliseconds |

### Split Point Selection

The `--rpc-num-host-layers` parameter controls how many layers run on the host vs VM:
- **Lower values** (e.g., 8): More work on VM, less on host
- **Higher values** (e.g., 24): More work on host, less on VM
- **Default** (16): Balanced split for 32-layer models

Choose based on:
- Relative GPU memory/capabilities of host vs VM
- Network latency between machines
- Target inference speed vs memory distribution

## Troubleshooting

### VM Worker Issues

**Problem**: VM worker fails to start
```
RPC initialization failed: Address already in use
```
**Solution**: Ensure the port is not already in use. Change `--vm-address` to use a different port.

**Problem**: CUDA out of memory on VM
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce cache size or decrease `--rpc-num-host-layers` to move fewer layers to VM.

### Host Issues

**Problem**: Host fails to connect to VM
```
RPC sync call to vm.run_decode_step failed: ...
```
**Solution**:
1. Ensure VM worker is running and accessible
2. Check firewall settings between host and VM
3. Verify `--rpc-host-address` and `--rpc-vm-address` match

**Problem**: Model loading fails in RPC mode
```
Model loading failed: ...
```
**Solution**:
1. Ensure both host and VM have access to the checkpoint path
2. Verify the checkpoint exists and is readable on both machines

### Performance Issues

**Problem**: RPC mode is slower than local mode
**Solutions**:
1. Check network latency between host and VM (should be <10ms for same LAN)
2. Try different split points with `--rpc-num-host-layers`
3. Ensure both machines have similar GPU capabilities
4. Monitor GPU utilization on both machines during inference

## Implementation Details

### Tied Embeddings

The model uses tied embeddings (output layer shares weights with input embeddings). For RPC mode:
- Host retains original embeddings.weight
- VM receives a copy of embeddings.weight (~40MB for 32000 × 4096)
- This ensures identical logits without additional RPC calls

### KV Cache Management

- Each worker creates and manages its own KV caches
- Host: caches for layers [0, num_host_layers)
- VM: caches for layers [num_host_layers, n_layer) + fast layers
- No cache transfer over RPC (reduces bandwidth)

### RPC Payload

Per decode step, the host sends:
```python
@dataclass
class RPCDecodePayload:
    x: Tensor              # (1, 1, dim) - hidden states after host layers
    input_pos: Tensor      # (1,) - current position
    audio_masks: Tensor    # Semantic filtering mask
    audio_parts: Tensor    # Audio embedding parts
    temperature: Tensor    # Sampling params
    top_p: Tensor
    top_k: int
    semantic_logit_bias: Tensor  # (1, 1, vocab_size)
    previous_tokens: Tensor # (num_codebooks+1, RAS_WIN_SIZE) for RAS
```

Approximate size: ~16-32KB per decode step (depending on hidden dim).

### Backward Compatibility

RPC mode is **opt-in** via the `--rpc-enable` flag. Without this flag:
- All existing functionality works unchanged
- No performance impact
- No additional dependencies

## Limitations

1. **torch.compile disabled**: RPC mode runs in eager mode (torch.compile not yet supported for distributed execution)
2. **Batch size 1**: Current implementation only supports batch_size=1
3. **Fixed split point**: Split point must be set at startup (cannot be changed dynamically)
4. **Single VM**: Only one VM worker supported per host

## Future Enhancements

Potential improvements for future versions:
1. Support for `torch.compile` on individual workers
2. Dynamic split point tuning based on model size and memory
3. Multi-GPU VM support for fast AR parallelization
4. RPC payload compression for lower bandwidth
5. Support for multiple VM workers per host

## References

- Main implementation: `fish_speech/models/text2semantic/inference_rpc.py`
- Host runner: `fish_speech/models/text2semantic/rpc_host.py`
- VM runner: `fish_speech/models/text2semantic/rpc_vm.py`
- VM launcher: `tools/launch_rpc_vm.py`
- Host launcher: `tools/launch_rpc_host.sh`
