# Split Mode Tools

This directory contains tools for running Fish Speech in split mode, where the slow AR (4B parameter) model runs on a remote GPU and the fast AR (400M parameter) model runs locally.

## Tools

### `vm_worker.py`

Runs on the remote VM (e.g., with a 1080 Ti GPU) and executes only the slow AR component.

```bash
python tools/split/vm_worker.py \
    --checkpoint-path checkpoints/s2-pro \
    --device cuda:0 \
    --port 50061 \
    --dtype float16
```

**Options**:
- `--checkpoint-path`: Path to model checkpoint
- `--device`: Device to run on (default: cuda:0)
- `--port`: Port to listen on (default: 50061)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--dtype`: Data type (float16 or float32, default: float16)

### `benchmark_latency.py`

Tests TCP round-trip latency between host and VM to verify network performance.

```bash
# On VM (server):
python tools/split/benchmark_latency.py --mode server --port 50061

# On host (client):
python tools/split/benchmark_latency.py --mode client \
    --server-addr 192.168.122.10 --port 50061 \
    --iterations 1000 --tensor-size 4096
```

**Acceptance criteria**: Median round-trip < 1ms for local network.

## Quick Start

1. **Start VM worker** (on remote machine with 1080 Ti):
   ```bash
   python tools/split/vm_worker.py --checkpoint-path checkpoints/s2-pro
   ```

2. **Test latency** (from host):
   ```bash
   python tools/split/benchmark_latency.py --mode client \
       --server-addr 192.168.122.10
   ```

3. **Run inference** (from host):
   ```bash
   python fish_speech/models/text2semantic/inference.py \
       --text "Hello, world!" \
       --split-mode slow_full_remote \
       --remote-worker 192.168.122.10:50061
   ```

## Hardware Requirements

### VM (Remote)
- GPU: 11GB+ VRAM (e.g., GTX 1080 Ti, RTX 2080 Ti)
- Network: Stable connection to host
- Software: Python 3.10+, PyTorch with CUDA

### Host (Local)
- GPU: 4GB+ VRAM (e.g., RTX 5060 Ti, RTX 3060)
- Network: Low latency to VM (< 1ms recommended)
- Software: Fish Speech with split mode support

## Troubleshooting

See [Split Mode Documentation](../../docs/en/split_mode.md) for detailed troubleshooting.

### Common Issues

1. **Connection refused**: Check firewall and port accessibility
2. **Timeout errors**: Verify network stability and GPU memory on VM
3. **Poor quality**: Ensure fp16 on both sides (not mixed fp16/bf16)
4. **Slow generation**: Profile network latency with benchmark tool

## Architecture

```
Host (Local)                    VM (Remote)
──────────────────────────────────────────────
Fast AR (400M)          ──────▶  Slow AR (4B)
DAC Codec                        RAS Sampling
                                 Token Embeddings
                                  Slow Layers
```

For more details, see the [full documentation](../../docs/en/split_mode.md).
