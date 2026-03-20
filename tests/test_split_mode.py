#!/usr/bin/env python3
"""
Tests for Split Mode functionality

Run with: python -m pytest tests/test_split_mode.py -v
"""

import socket
import threading
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from fish_speech.models.text2semantic.split_protocol import (
    Op,
    bytes_to_tensor,
    build_close_session,
    build_start_session,
    build_step,
    decode_message,
    encode_message,
    dtype_from_string,
    get_tensor_info,
    tensor_to_bytes,
)
from fish_speech.models.text2semantic.backends import (
    FastARBackend,
    LocalSlowARBackend,
)


class TestProtocolSerialization:
    """Test protocol encoding/decoding."""

    def test_encode_decode_message(self):
        """Test basic message encoding and decoding."""
        header = {"op": Op.START_SESSION, "session_id": 123}
        payload = b"test payload data"

        encoded = encode_message(header, payload)
        decoded_header, decoded_payload = decode_message(encoded)

        assert decoded_header["op"] == Op.START_SESSION
        assert decoded_header["session_id"] == 123
        assert decoded_payload == payload

    def test_encode_decode_message_no_payload(self):
        """Test message encoding without payload."""
        header = {"op": Op.CLOSE_SESSION, "session_id": 456}

        encoded = encode_message(header, None)
        decoded_header, decoded_payload = decode_message(encoded)

        assert decoded_header["op"] == Op.CLOSE_SESSION
        assert decoded_header["session_id"] == 456
        assert decoded_payload == b""

    def test_tensor_serialization_roundtrip(self):
        """Test tensor serialization and reconstruction."""
        # Test various dtypes and shapes
        test_cases = [
            (torch.randn(10, 20), "float32", (10, 20)),
            (torch.randn(5, 5, 5), "float16", (5, 5, 5)),
            (torch.randint(0, 1000, (3, 4)), "int32", (3, 4)),
        ]

        for original_tensor, dtype_str, shape in test_cases:
            if dtype_str == "float16":
                original_tensor = original_tensor.to(torch.float16)

            info = get_tensor_info(original_tensor)
            payload = tensor_to_bytes(original_tensor)

            reconstructed = bytes_to_tensor(
                payload, dtype_from_string(info["dtype"]), tuple(info["shape"])
            )

            assert reconstructed.shape == original_tensor.shape
            assert reconstructed.dtype == original_tensor.dtype
            assert torch.allclose(reconstructed, original_tensor, atol=1e-5)

    def test_build_start_session(self):
        """Test START_SESSION message building."""
        header, payload = build_start_session(
            session_id=1,
            prompt_tokens=[1, 2, 3, 4, 5],
            sampling_config={"temperature": 0.7, "top_p": 0.9, "top_k": 30},
            input_seq_len=5,
        )

        assert header["op"] == Op.START_SESSION
        assert header["session_id"] == 1
        assert header["prompt_tokens"] == [1, 2, 3, 4, 5]
        assert header["sampling_config"]["temperature"] == 0.7
        assert payload is None

    def test_build_step(self):
        """Test STEP message building."""
        header, payload = build_step(
            session_id=1, prev_codebooks=[100, 200, 300], input_pos=42
        )

        assert header["op"] == Op.STEP
        assert header["session_id"] == 1
        assert header["prev_codebooks"] == [100, 200, 300]
        assert header["input_pos"] == 42
        assert payload is None

    def test_build_close_session(self):
        """Test CLOSE_SESSION message building."""
        header, payload = build_close_session(session_id=999)

        assert header["op"] == Op.CLOSE_SESSION
        assert header["session_id"] == 999
        assert payload is None


class TestBackends:
    """Test backend implementations."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock DualARTransformer model."""
        model = Mock()
        model.config = Mock()
        model.config.num_codebooks = 10
        model.config.codebook_size = 160
        model.config.semantic_begin_id = 0
        model.config.semantic_end_id = 160
        model.device = "cpu"

        # Mock tokenizer
        model.tokenizer = Mock()
        model.tokenizer.get_token_id = Mock(return_value=2)

        return model

    def test_fast_ar_backend_init(self, mock_model):
        """Test FastARBackend initialization."""
        backend = FastARBackend(mock_model, "cpu")

        assert backend.model == mock_model
        assert backend.device == "cpu"

    def test_local_slow_ar_backend_init(self, mock_model):
        """Test LocalSlowARBackend initialization."""
        backend = LocalSlowARBackend(mock_model, "cpu")

        assert backend.model == mock_model
        assert backend.device == "cpu"
        assert backend._session_counter == 0
        assert len(backend._sessions) == 0


class TestSocketHelpers:
    """Test socket helper functions."""

    @pytest.fixture
    def echo_server(self):
        """Create a simple echo server for testing."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))  # Bind to random port
        server_sock.listen(1)

        port = server_sock.getsockname()[1]

        def echo_handler():
            conn, _ = server_sock.accept()
            try:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    conn.sendall(data)
            finally:
                conn.close()

        thread = threading.Thread(target=echo_handler, daemon=True)
        thread.start()

        time.sleep(0.1)  # Give server time to start

        yield ("127.0.0.1", port)

        server_sock.close()

    def test_send_recv_message(self, echo_server):
        """Test sending and receiving messages through socket."""
        from fish_speech.models.text2semantic.split_protocol import (
            recv_message,
            send_message,
        )

        host, port = echo_server

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        # Send a message
        header = {"op": Op.START_SESSION, "session_id": 123}
        payload = b"test data"
        send_message(sock, header, payload)

        # Receive echoed response
        response = recv_message(sock, timeout=1.0)

        sock.close()

        assert response is not None
        decoded_header, decoded_payload = decode_message(response)
        assert decoded_header["session_id"] == 123
        assert decoded_payload == payload


class TestLatencyBenchmark:
    """Test latency benchmark tool."""

    def test_tensor_size_calculation(self):
        """Test that tensor size calculations are correct."""
        # Typical hidden state sizes
        sizes = [
            (4096, "float16"),  # Small model
            (4096, "float32"),  # Small model fp32
            (5120, "float16"),  # Larger model
        ]

        for size, dtype in sizes:
            tensor = torch.randn(1, 1, size, dtype=getattr(torch, dtype))
            info = get_tensor_info(tensor)

            expected_bytes = tensor.numel() * tensor.element_size()
            assert info["num_bytes"] == expected_bytes

            # Verify roundtrip works
            payload = tensor_to_bytes(tensor)
            assert len(payload) == expected_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
