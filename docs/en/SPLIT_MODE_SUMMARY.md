# Split Mode Implementation Summary

This document summarizes the split mode implementation for Fish Speech, enabling distributed inference across two GPUs.

## Files Created

### Core Implementation

1. **`fish_speech/models/text2semantic/split_protocol.py`**
   - Wire protocol definition for TCP communication
   - Message opcodes (START_SESSION, STEP, CLOSE_SESSION, ERROR)
   - Tensor serialization/deserialization functions
   - Socket helper functions (send_message, recv_message)
   - Message builders and parsers for all message types

2. **`fish_speech/models/text2semantic/backends.py`**
   - `SlowARBackend`: Abstract base class for slow AR backends
   - `FastARBackend`: Local fast AR backend (always runs locally)
   - `LocalSlowARBackend`: Local implementation for testing/non-split mode
   - `RemoteSlowARBackend`: Client-side remote backend for split mode

### VM Worker

3. **`tools/split/vm_worker.py`**
   - Standalone worker process for VM
   - Loads only slow AR components (4B params)
   - Handles START_SESSION, STEP, CLOSE_SESSION requests
   - Implements RAS sampling on VM side
   - Multi-threaded to handle multiple clients

### Tools

4. **`tools/split/benchmark_latency.py`**
   - TCP latency benchmark tool
   - Tests round-trip time with tensor payloads
   - Reports min, max, mean, median, p95, p99 latencies
   - Acceptance criteria: Median < 1ms

### Configuration

5. **`fish_speech/configs/split_mode.yaml`**
   - Split mode configuration template
   - Settings for remote worker address, dtype, etc.

### Tests

6. **`tests/test_split_mode.py`**
   - Protocol serialization tests
   - Backend initialization tests
   - Socket helper tests
   - Tensor roundtrip tests
   - Can be run with: `pytest tests/test_split_mode.py -v`

### Documentation

7. **`docs/en/split_mode.md`**
   - Comprehensive documentation for split mode
   - Hardware requirements
   - Quick start guide
   - Troubleshooting guide
   - Performance expectations

8. **`tools/split/README.md`**
   - Quick reference for split mode tools
   - Command examples
   - Common issues and solutions

## Files Modified

### Core Inference

9. **`fish_speech/models/text2semantic/inference.py`**
   - Added `slow_step()` function (extracted from decode_one_token_ar)
   - Added `fast_step()` function (extracted from decode_one_token_ar)
   - Added `decode_one_token_with_backends()` for split mode
   - Added `init_model_with_split()` for backend initialization
   - Added CLI arguments: `--split-mode`, `--remote-worker`, `--split-dtype`
   - Modified `main()` to support split mode initialization
   - Split mode validation and configuration handling

### Model Architecture

10. **`fish_speech/models/text2semantic/llama.py`**
    - Added `load_slow_only()` classmethod to DualARTransformer
    - Added `load_fast_only()` classmethod to DualARTransformer
    - These enable loading only the required components for each side

### API Server

11. **`tools/server/api_utils.py`**
    - Added split mode CLI arguments to `parse_args()`
    - `--split-mode`, `--remote-worker`, `--split-dtype`

12. **`tools/server/model_manager.py`**
    - Added split mode parameters to `__init__`
    - Modified `load_llama_model()` to handle split mode
    - Added split mode validation and logging

13. **`tools/api_server.py`**
    - Updated `initialize_app()` to pass split mode args to ModelManager

### Dependencies

14. **`pyproject.toml`**
    - Added `msgpack` dependency for protocol serialization

## Architecture Overview

```
Host Machine (Local)                    VM (Remote)
────────────────────────────────────────────────────────
FastARBackend                          VMWorker
├─ Fast AR (400M)                      ├─ Slow AR (4B)
│  ├─ 4 fast layers                   │  ├─ 36 slow layers
│  ├─ Fast embeddings                 │  ├─ Token embeddings
│  └─ Fast LM head                    │  └─ Slow LM head
│                                       │
RemoteSlowARBackend                    ├─ RAS Sampling
├─ TCP client                          ├─ Semantic token generation
├─ Session management                  └─ Hidden state extraction
└─ Protocol handling
                                       │
DAC Codec                              └─ TCP server
├─ Encoder/Decoder
└─ VQ encoding
```

## Message Flow

1. **START_SESSION**: Host → VM with prompt tokens
2. **START_SESSION_RESULT**: VM → Host with first semantic token + hidden state
3. **STEP**: Host → VM with previous codebooks
4. **STEP_RESULT**: VM → Host with semantic token + hidden state
5. Repeat steps 3-4 until EOS
6. **CLOSE_SESSION**: Host → VM to cleanup

## Key Design Decisions

1. **fp16 everywhere**: 1080 Ti doesn't support bf16, so both sides use fp16
2. **No torch.compile**: Cross-process + compile is problematic, always disabled in split mode
3. **Single session**: v1 supports one request at a time
4. **RAS on VM**: Repetition Aware Sampling requires access to logits
5. **Msgpack for headers**: Efficient binary serialization for metadata
6. **Raw bytes for tensors**: Direct memory transmission without overhead

## Usage Examples

### VM Worker (on remote machine)
```bash
python tools/split/vm_worker.py \
    --checkpoint-path checkpoints/s2-pro \
    --device cuda:0 \
    --port 50061 \
    --dtype float16
```

### Inference (on host)
```bash
python fish_speech/models/text2semantic/inference.py \
    --text "Hello, world!" \
    --split-mode slow_full_remote \
    --remote-worker 192.168.122.10:50061 \
    --split-dtype float16
```

### API Server (on host)
```bash
python tools/api_server.py \
    --llama-checkpoint-path checkpoints/s2-pro \
    --split-mode slow_full_remote \
    --remote-worker 192.168.122.10:50061 \
    --listen 0.0.0.0:8080
```

## Verification Steps

1. **Test network latency**:
   ```bash
   python tools/split/benchmark_latency.py --mode client \
       --server-addr 192.168.122.10
   ```
   Expected: Median < 1ms

2. **Test protocol serialization**:
   ```bash
   pytest tests/test_split_mode.py::TestProtocolSerialization -v
   ```

3. **Test local backend**:
   ```bash
   pytest tests/test_split_mode.py::TestBackends -v
   ```

4. **End-to-end test**: Run inference with split mode and verify audio quality

## Limitations (v1)

- Single session at a time (no concurrent requests)
- No batching support
- Requires stable, low-latency network (< 1ms recommended)
- No torch.compile support in split mode
- fp16 only (no bf16 support for 1080 Ti)

## Future Improvements

- Multi-session support with connection pooling
- Batch processing for multiple requests
- Dynamic load balancing across multiple VMs
- Compression for tensor payloads
- Automatic failover and reconnection
- Support for more GPU architectures
- Performance profiling and optimization

## Troubleshooting

See `docs/en/split_mode.md` for detailed troubleshooting guide.

Common issues:
- Connection refused → Check firewall and port
- Timeout errors → Check network stability and GPU memory
- Poor audio quality → Verify fp16 on both sides, check latency
- Slow generation → Profile network and GPU utilization

## Dependencies

- `msgpack`: Protocol serialization
- `torch`: Model inference
- `numpy`: Tensor operations
- Existing Fish Speech dependencies

## Testing

Run the test suite:
```bash
pytest tests/test_split_mode.py -v
```

Run specific test categories:
```bash
# Protocol tests
pytest tests/test_split_mode.py::TestProtocolSerialization -v

# Backend tests
pytest tests/test_split_mode.py::TestBackends -v

# Socket tests
pytest tests/test_split_mode.py::TestSocketHelpers -v
```

## Performance

Expected performance metrics:
- Network latency: < 1ms (median)
- Token generation: 10-30 tokens/sec
- Host VRAM: ~4GB
- VM VRAM: ~11GB
- Total VRAM: ~15GB (vs ~24GB for local)

## License

Same as Fish Speech main project.
