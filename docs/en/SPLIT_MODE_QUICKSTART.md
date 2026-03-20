# Split Mode Quick Start Guide

This guide will help you get started with split mode for Fish Speech.

## What is Split Mode?

Split mode allows you to run Fish Speech inference across two GPUs:
- **Remote VM (1080 Ti, 11GB)**: Runs the 4B parameter slow AR model
- **Local host (5060 Ti, 16GB)**: Runs the 400M parameter fast AR model + DAC codec

This reduces local VRAM requirements from ~24GB to ~4GB.

## Prerequisites

1. Two machines connected via low-latency network (< 1ms round-trip)
2. VM with 11GB+ GPU (e.g., GTX 1080 Ti)
3. Host with 4GB+ GPU (e.g., RTX 5060 Ti)
4. Model weights downloaded: `checkpoints/s2-pro/`

## Step 1: Start the VM Worker

On the **remote VM** (with the 1080 Ti):

```bash
# Activate conda environment
conda activate fish-speech

# Start the worker
python tools/split/vm_worker.py \
    --checkpoint-path checkpoints/s2-pro \
    --device cuda:0 \
    --port 50061 \
    --dtype float16
```

Expected output:
```
INFO - Loading slow AR model from checkpoints/s2-pro
INFO - Deleting fast AR components to save VRAM
INFO - Slow AR model loaded on cuda:0 with dtype torch.float16
INFO - VM worker listening on 0.0.0.0:50061
Ready to accept connections from host
```

## Step 2: Test Network Latency

On the **host machine**:

```bash
# Activate conda environment
conda activate fish-speech

# Run latency benchmark
python tools/split/benchmark_latency.py --mode client \
    --server-addr 192.168.122.10 \
    --port 50061 \
    --iterations 1000
```

Replace `192.168.122.10` with your VM's IP address.

Expected output:
```
Latency Benchmark Results
==============================
Iterations:       1000
Tensor size:      4096 elements (32768 bytes)
Min latency:      0.234 ms
Max latency:      2.145 ms
Mean latency:     0.456 ms
Median latency:   0.412 ms
95th percentile:  0.823 ms
99th percentile:  1.234 ms
==============================
✓ PASS: Median latency < 1ms
```

## Step 3: Run Inference with Split Mode

On the **host machine**:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "Hello, this is a test of split mode." \
    --split-mode slow_full_remote \
    --remote-worker 192.168.122.10:50061 \
    --split-dtype float16 \
    --checkpoint-path checkpoints/s2-pro \
    --output output_split_test.wav
```

Expected output:
```
INFO - Loading fast AR only on cuda (split mode)
INFO - Using remote slow AR backend at 192.168.122.10:50061
INFO - Connected to remote slow AR worker
INFO - Split into 1 turns, grouped into 1 batches
INFO - Encoded prompt shape: torch.Size([11, 3])
INFO - Generated 23 tokens in 2.34 seconds, 9.83 tokens/sec
INFO - Saved audio to output_split_test.wav
```

## Step 4: Run API Server with Split Mode (Optional)

On the **host machine**:

```bash
python tools/api_server.py \
    --llama-checkpoint-path checkpoints/s2-pro \
    --split-mode slow_full_remote \
    --remote-worker 192.168.122.10:50061 \
    --split-dtype float16 \
    --listen 0.0.0.0:8080
```

Then test with curl:

```bash
curl -X POST http://localhost:8080/v1/tts \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Hello from split mode API!",
        "references": [],
        "format": "wav"
    }' \
    --output test_api.wav
```

## Troubleshooting

### "Connection refused"
- Check VM worker is running
- Check firewall allows port 50061
- Verify IP address is correct

### "Timeout"
- Check VM has enough GPU memory
- Verify network is stable
- Check VM logs for errors

### Poor audio quality
- Ensure fp16 on both sides (not mixed fp16/bf16)
- Check network latency is < 1ms
- Verify same model checkpoint on both sides

### Slow generation
- Profile network: `python tools/split/benchmark_latency.py`
- Check GPU utilization on both machines
- Ensure no other GPU tasks are running

## Performance Expectations

| Metric | Expected |
|--------|----------|
| Network latency | < 1ms (median) |
| Token generation | 10-30 tokens/sec |
| Host VRAM usage | ~4GB |
| VM VRAM usage | ~11GB |

## Next Steps

- Read full documentation: `docs/en/split_mode.md`
- See implementation details: `docs/en/SPLIT_MODE_SUMMARY.md`
- Run tests: `pytest tests/test_split_mode.py -v`
- Check tools: `tools/split/README.md`

## Support

For issues or questions:
1. Check `docs/en/split_mode.md` troubleshooting section
2. Review VM logs for errors
3. Test with latency benchmark tool
4. Verify network connectivity and GPU availability
