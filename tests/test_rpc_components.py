"""
Basic tests for RPC-based layer offload components.

These tests verify the structure and basic functionality of the RPC components
without actually running RPC (which requires two separate processes).
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock

from fish_speech.models.text2semantic.inference_rpc import (
    RPCDecodePayload,
    RPCDecodeResponse,
    multinomial_sample_one_no_sync,
    logits_to_probs,
    sample,
)


def test_rpc_decode_payload_creation():
    """Test that RPCDecodePayload can be created with required fields."""
    payload = RPCDecodePayload(
        x=torch.randn(1, 1, 4096),
        input_pos=torch.tensor([42]),
        audio_masks=torch.zeros(1, dtype=torch.bool),
        audio_parts=torch.zeros(1, 1, 512),
        temperature=torch.tensor(0.7),
        top_p=torch.tensor(0.9),
        top_k=30,
        semantic_logit_bias=torch.zeros(1, 1, 32000),
        previous_tokens=torch.zeros(10, 10, dtype=torch.int),
    )

    assert payload.x.shape == (1, 1, 4096)
    assert payload.input_pos.item() == 42
    assert payload.top_k == 30


def test_rpc_decode_response_creation():
    """Test that RPCDecodeResponse can be created with required fields."""
    response = RPCDecodeResponse(
        codebooks=torch.randint(0, 160, (10, 1)),
        success=True,
    )

    assert response.success is True
    assert response.codebooks.shape == (10, 1)
    assert response.error_msg is None


def test_rpc_decode_response_failure():
    """Test that RPCDecodeResponse can represent failures."""
    response = RPCDecodeResponse(
        codebooks=None,
        success=False,
        error_msg="Test error message",
    )

    assert response.success is False
    assert response.codebooks is None
    assert response.error_msg == "Test error message"


def test_multinomial_sample_one_no_sync():
    """Test multinomial sampling function."""
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    sample = multinomial_sample_one_no_sync(probs)

    assert sample.shape == (1, 1)
    assert sample.dtype == torch.int
    # Sample should be one of the valid indices
    assert 0 <= sample.item() < 4


def test_logits_to_probs():
    """Test logits to probabilities conversion."""
    logits = torch.randn(1, 1, 50)
    temperature = torch.tensor(1.0)
    top_p = torch.tensor(0.9)
    top_k = 30

    probs = logits_to_probs(logits, temperature, top_p, top_k)

    assert probs.shape == (1, 50)
    # Probabilities should sum to approximately 1
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    # All probabilities should be non-negative
    assert (probs >= 0).all()


def test_sample():
    """Test sampling from logits."""
    logits = torch.randn(1, 1, 50)
    temperature = torch.tensor(1.0)
    top_p = torch.tensor(0.9)
    top_k = 30

    idx, probs = sample(logits, temperature, top_p, top_k)

    assert idx.shape == (1, 1)
    assert probs.shape == (1, 50)
    assert idx.dtype == torch.int


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rpc_payload_cuda():
    """Test that RPC payload works with CUDA tensors."""
    device = torch.device("cuda")

    payload = RPCDecodePayload(
        x=torch.randn(1, 1, 4096, device=device),
        input_pos=torch.tensor([42], device=device),
        audio_masks=torch.zeros(1, dtype=torch.bool, device=device),
        audio_parts=torch.zeros(1, 1, 512, device=device),
        temperature=torch.tensor(0.7, device=device),
        top_p=torch.tensor(0.9, device=device),
        top_k=30,
        semantic_logit_bias=torch.zeros(1, 1, 32000, device=device),
        previous_tokens=torch.zeros(10, 10, dtype=torch.int, device=device),
    )

    assert payload.x.device.type == "cuda"
    assert payload.input_pos.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
