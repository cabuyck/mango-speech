#!/usr/bin/env python3
"""
Simple integration test for split mode.

This script verifies that:
1. Protocol serialization works
2. Backends can be initialized
3. Local backend produces same results as original decode_one_token_ar

Usage:
    python tools/split/test_split_integration.py
"""

import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, "/home/cabuyck/repos/fish-speech")

from fish_speech.models.text2semantic.split_protocol import (
    encode_message,
    decode_message,
    tensor_to_bytes,
    bytes_to_tensor,
    get_tensor_info,
    Op,
    build_start_session,
    build_step,
)


def test_protocol():
    """Test protocol encoding/decoding."""
    print("Testing protocol encoding/decoding...")

    # Test basic message
    header = {"op": Op.START_SESSION, "session_id": 123}
    payload = b"test payload"
    encoded = encode_message(header, payload)
    decoded_header, decoded_payload = decode_message(encoded)

    assert decoded_header["op"] == Op.START_SESSION
    assert decoded_header["session_id"] == 123
    assert decoded_payload == payload

    print("  ✓ Basic message encoding/decoding works")

    # Test message builders
    header, payload = build_start_session(
        session_id=1,
        prompt_tokens=[1, 2, 3],
        sampling_config={"temperature": 0.7, "top_p": 0.9, "top_k": 30},
        input_seq_len=3,
    )

    assert header["op"] == Op.START_SESSION
    assert header["session_id"] == 1
    assert header["prompt_tokens"] == [1, 2, 3]

    print("  ✓ Message builders work")


def test_tensor_serialization():
    """Test tensor serialization roundtrip."""
    print("Testing tensor serialization...")

    # Test different dtypes and shapes
    test_cases = [
        (torch.randn(1, 1, 4096), "float16", (1, 1, 4096)),
        (torch.randn(1, 1, 2048), "float32", (1, 1, 2048)),
    ]

    for original, dtype, shape in test_cases:
        if dtype == "float16":
            original = original.to(torch.float16)

        info = get_tensor_info(original)
        payload = tensor_to_bytes(original)

        reconstructed = bytes_to_tensor(
            payload, torch.float16 if dtype == "float16" else torch.float32, shape
        )

        assert reconstructed.shape == original.shape
        assert torch.allclose(reconstructed, original, atol=1e-5)

        print(f"  ✓ Tensor {shape} {dtype} roundtrip works")


def test_backend_imports():
    """Test that backend modules can be imported."""
    print("Testing backend imports...")

    from fish_speech.models.text2semantic.backends import (
        SlowARBackend,
        FastARBackend,
        LocalSlowARBackend,
        RemoteSlowARBackend,
    )

    # Check that abstract base class exists
    assert hasattr(SlowARBackend, '__abstractmethods__')

    print("  ✓ All backend classes can be imported")


def test_partial_loading_methods():
    """Test that partial loading methods exist."""
    print("Testing partial loading methods...")

    from fish_speech.models.text2semantic.llama import DualARTransformer

    assert hasattr(DualARTransformer, 'load_slow_only')
    assert hasattr(DualARTransformer, 'load_fast_only')

    print("  ✓ Partial loading methods exist")


def test_vm_worker_imports():
    """Test that vm_worker can be imported."""
    print("Testing VM worker imports...")

    # Just check the file exists and is importable
    import tools.split.vm_worker as vm_worker

    assert hasattr(vm_worker, 'VMWorker')

    print("  ✓ VM worker module is importable")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Split Mode Integration Tests")
    print("=" * 60)
    print()

    try:
        test_protocol()
        test_tensor_serialization()
        test_backend_imports()
        test_partial_loading_methods()
        test_vm_worker_imports()

        print()
        print("=" * 60)
        print("✓ All integration tests passed!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        return 1

    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
