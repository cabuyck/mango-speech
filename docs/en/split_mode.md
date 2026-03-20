# Split Mode Documentation

Split mode enables distributed inference across two GPUs in different environments by splitting the Dual-AR architecture. The slow AR (4B parameters) runs on a remote GPU while the fast AR (400M parameters) and DAC codec run locally.

## Use Cases

- **Hardware limitations**: Your local GPU doesn't have enough VRAM for the full model (~24GB)
- **Remote acceleration**: You have access to a remote GPU with more VRAM
- **Cost optimization**: Use a smaller local GPU for fast AR and a remote GPU for slow AR

## Architecture

```
Host Machine (e.g., 5060 Ti, 16GB)          VM/Remote (e.g., 1080 Ti, 11GB)
┌─────────────────────────────┐           ┌─────────────────────────────┐
│  Fast AR (400M params)      │           │  Slow AR (4B params)        │
│  - 4 fast layers            │           │  - 36 slow layers           │
│  - Fast embeddings          │           │  - Token embeddings         │
│  - Fast LM head             │           │  - Codebook embeddings      │
│  - KV caches (10 codebooks) │           │  - Slow LM head             │
│                             │           │  - KV caches (seq_len)      │
│  DAC Codec                  │           │                             │
│  - Encoder/Decoder          │◄───TCP────▶│  RAS Sampling               │
│  - VQ encoding              │  hidden   │  - Semantic token gen       │
│                             │  state    │                             │
└─────────────────────────────┘           └─────────────────────────────┘
```

## Hardware Requirements

### Host (Local Machine)
- **GPU**: Any GPU with ~4GB VRAM (e.g., RTX 5060 Ti, RTX 3060)
- **Network**: Low latency connection to VM (< 1ms round-trip recommended)
- **Software**: Fish Speech with split mode enabled

### VM (Remote Machine)
- **GPU**: At least 11GB VRAM (e.g., GTX 1080 Ti, RTX 2080 Ti)
- **Network**: Stable connection to host
- **Software**: Python 3.10+, PyTorch with CUDA support
- **Note**: 1080 Ti doesn't support bf16, so fp16 is used

## Quick Start

### 1. Start the VM Worker

On the remote VM (with the 1080 Ti):

```bash
# Download model weights (if not already present)
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro

# Start the worker
python tools/split/vm_worker.py \
    --checkpoint-path checkpoints/s2-pro \
    --device cuda:0 \
    --port 50061 \
    --dtype float16
```

The worker will listen on port 50061 for connections from the host.

### 2. Test Latency

On the host machine, test network latency:

```bash
# Start server on VM (in one terminal)
python tools/split/benchmark_latency.py --mode server --port 50061

# Run client on host (in another terminal)
python tools/split/benchmark_latency.py --mode client \
    --server-addr 192.168.122.10 --port 50061 --iterations 1000
```

**Acceptance criteria**: Median round-trip < 1ms for local network.

### 3. Run Inference with Split Mode

On the host machine:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "Hello, this is a test." \
    --split-mode slow_full_remote \
    --remote-worker 192.168.122.10:50061 \
    --split-dtype float16 \
    --checkpoint-path checkpoints/s2-pro
```

### 4. Run API Server with Split Mode

```bash
python tools/api_server.py \
    --llama-checkpoint-path checkpoints/s2-pro \
    --split-mode slow_full_remote \
    --remote-worker 192.168.122.10:50061 \
    --split-dtype float16 \
    --listen 0.0.0.0:8080
```

## Configuration

Split mode can be configured via `fish_speech/configs/split_mode.yaml`:

```yaml
split_mode:
  enabled: true
  mode: "slow_full_remote"
  remote_worker: "192.168.122.10:50061"
  dtype: "float16"
  compile: false  # Always false in split mode
  max_concurrent_requests: 1

compile: false
precision: "fp16"
```

## Troubleshooting

### Connection Issues

**Problem**: `ConnectionError: Connection refused`

**Solution**:
- Check that the VM worker is running
- Verify the firewall allows TCP on port 50061
- Check network connectivity: `ping <vm-ip>` and `telnet <vm-ip> 50061`

### Timeout Issues

**Problem**: `TimeoutError: Socket receive timeout after 30s`

**Solution**:
- Check VM logs for errors
- Verify VM has enough GPU memory
- Try reducing `max_new_tokens` to decrease generation time
- Check network stability

### Out of Memory on VM

**Problem**: CUDA out of memory on VM

**Solution**:
- Ensure no other processes are using GPU memory
- Reduce `max_seq_len` if customizing the model
- Verify fp16 mode is enabled (not fp32)

### Poor Audio Quality

**Problem**: Generated audio quality is worse than local mode

**Solution**:
- Check that fp16 is used on both sides (dtype mismatch can cause issues)
- Verify network latency is < 1ms median
- Check that both host and VM are using the same model checkpoint

### Performance Issues

**Problem**: Generation is slower than expected

**Solution**:
- Profile network latency with `benchmark_latency.py`
- Check GPU utilization on both host and VM
- Verify that the VM is not running other GPU tasks
- Ensure TCP window scaling is enabled on both systems

## Performance Expectations

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| Network latency | < 1ms (median) | Local network between host and VM |
| Token generation | 10-30 tokens/sec | Depends on hardware and network |
| VRAM usage (host) | ~4GB | Fast AR + DAC codec only |
| VRAM usage (VM) | ~11GB | Full slow AR model |
| Total VRAM | ~15GB | Split across two GPUs |

## Limitations

- **Single session**: Only one generation request at a time in v1
- **No batching**: Batch size is fixed at 1
- **Network dependency**: Requires stable, low-latency network
- **No compile**: `torch.compile` is disabled in split mode
- **fp16 required**: 1080 Ti doesn't support bf16, so fp16 is used

## Advanced Topics

### Protocol Details

The wire protocol uses msgpack for headers and raw bytes for tensor data:

```
[4-byte header_len][msgpack header][8-byte payload_len][raw bytes]
```

Message types:
- `START_SESSION`: Initialize generation with prompt tokens
- `STEP`: Generate next semantic token
- `CLOSE_SESSION`: Clean up resources
- `ERROR`: Report errors from VM worker

### Session Management

Each generation request creates a session:
1. Host sends `START_SESSION` with prompt tokens
2. VM allocates KV cache and generates first semantic token
3. Host sends `STEP` requests with previous codebooks
4. VM returns semantic token + hidden state
5. Host uses fast AR to generate 10 codebooks
6. Repeat until EOS or max tokens
7. Host sends `CLOSE_SESSION`

### RAS Sampling

Repetition Aware Sampling (RAS) runs on the VM since it needs access to the logits. The VM maintains a rolling window of previous tokens and uses high-temperature sampling when repetitions are detected.

## Future Improvements

Potential enhancements for future versions:
- Multi-session support (concurrent requests)
- Batch processing
- Dynamic load balancing
- Automatic failover
- Compression for tensor payloads
- Support for more GPU architectures

## References

- [Dual-AR Architecture](https://github.com/fishaudio/fish-speech)
- [VM Worker Implementation](../../tools/split/vm_worker.py)
- [Protocol Definition](../../fish_speech/models/text2semantic/split_protocol.py)
