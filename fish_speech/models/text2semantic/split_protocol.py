"""
Split Mode Protocol

Defines the wire protocol for communication between host and VM in split mode.
The protocol splits the Dual-AR architecture, running slow AR on VM and fast AR locally.

Wire format:
    [4-byte header_len][msgpack header][8-byte payload_len][raw bytes payload]

Message types (opcodes):
    - START_SESSION: Initialize generation session with prompt tokens
    - START_SESSION_RESULT: Return initial semantic token, hidden state, EOS
    - STEP: Generate next semantic token given previous codebooks
    - STEP_RESULT: Return semantic token, hidden state, EOS, probs
    - CLOSE_SESSION: Clean up session resources
"""

import socket
import struct
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import msgpack
import torch
import numpy as np


class Op(StrEnum):
    """Message opcodes for split mode protocol."""

    START_SESSION = "start_session"
    START_SESSION_RESULT = "start_session_result"
    STEP = "step"
    STEP_RESULT = "step_result"
    CLOSE_SESSION = "close_session"
    ERROR = "error"


@dataclass
class SessionStartResult:
    """Result from START_SESSION operation."""

    semantic_token: int
    hidden_state: torch.Tensor  # Shape: (1, 1, hidden_dim)
    eos: bool


@dataclass
class StepResult:
    """Result from STEP operation."""

    semantic_token: int
    hidden_state: torch.Tensor  # Shape: (1, 1, hidden_dim)
    eos: bool
    probs: Optional[torch.Tensor] = None  # Optional probability distribution


@dataclass
class ErrorResponse:
    """Error response from VM worker."""

    error_type: str
    message: str


# ============================================================================
# Protocol Encoding/Decoding
# ============================================================================


def encode_message(header: dict, payload: Optional[bytes] = None) -> bytes:
    """
    Encode a message with header and payload.

    Args:
        header: Message header as dict (will be msgpack encoded)
        payload: Optional binary payload (e.g., tensor data)

    Returns:
        Encoded message bytes
    """
    header_bytes = msgpack.packb(header, use_bin_type=True)

    header_len = struct.pack("<I", len(header_bytes))
    payload_len = struct.pack("<Q", len(payload) if payload else 0)

    message = header_len + header_bytes + payload_len
    if payload:
        message += payload

    return message


def decode_message(data: bytes) -> tuple[dict, bytes]:
    """
    Decode a message into header and payload.

    Args:
        data: Raw message bytes

    Returns:
        Tuple of (header dict, payload bytes)
    """
    if len(data) < 4:
        raise ValueError("Message too short to read header length")

    header_len = struct.unpack("<I", data[:4])[0]

    if len(data) < 4 + header_len:
        raise ValueError("Message too short to read header")

    header = msgpack.unpackb(data[4 : 4 + header_len], raw=False)

    if len(data) < 4 + header_len + 8:
        raise ValueError("Message too short to read payload length")

    payload_len = struct.unpack("<Q", data[4 + header_len : 4 + header_len + 8])[0]

    payload = b""
    if payload_len > 0:
        if len(data) < 4 + header_len + 8 + payload_len:
            raise ValueError("Message too short to read payload")
        payload = data[4 + header_len + 8 : 4 + header_len + 8 + payload_len]

    return header, payload


# ============================================================================
# Tensor Serialization
# ============================================================================


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """
    Convert tensor to bytes for network transmission.

    Args:
        tensor: PyTorch tensor (will be made contiguous)

    Returns:
        Raw bytes representation
    """
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.detach().cpu().numpy().tobytes()


def bytes_to_tensor(
    data: bytes, dtype: torch.dtype, shape: tuple[int, ...]
) -> torch.Tensor:
    """
    Reconstruct tensor from bytes.

    Args:
        data: Raw bytes
        dtype: Target torch dtype
        shape: Target tensor shape

    Returns:
        Reconstructed tensor
    """
    # Map torch dtype to numpy dtype for frombuffer
    dtype_map = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "void",  # bfloat16 needs special handling
        torch.int32: "int32",
        torch.int64: "int64",
    }

    np_dtype = dtype_map.get(dtype, "float32")

    if dtype == torch.bfloat16:
        # bfloat16 special case: convert from int16
        array = np.frombuffer(data, dtype=np.int16)
        tensor = torch.from_numpy(array).view(torch.bfloat16)
    else:
        array = np.frombuffer(data, dtype=np_dtype)
        tensor = torch.from_numpy(array)

    return tensor.reshape(shape).clone()


def get_tensor_info(tensor: torch.Tensor) -> dict:
    """
    Extract metadata needed to reconstruct a tensor.

    Args:
        tensor: PyTorch tensor

    Returns:
        Dict with dtype, shape, and num_bytes
    """
    dtype_str = str(tensor.dtype).split(".")[-1]  # e.g., "torch.float16" -> "float16"

    return {
        "dtype": dtype_str,
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "num_bytes": tensor.numel() * tensor.element_size(),
    }


def dtype_from_string(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    return dtype_map.get(dtype_str, torch.float32)


# ============================================================================
# Message Builders
# ============================================================================


def build_start_session(
    session_id: int,
    prompt_tokens: list[int],  # Flattened prompt token IDs
    sampling_config: dict,
    input_seq_len: int,
) -> tuple[dict, Optional[bytes]]:
    """Build START_SESSION message."""
    header = {
        "op": Op.START_SESSION,
        "session_id": session_id,
        "sampling_config": sampling_config,
        "input_seq_len": input_seq_len,
    }
    # For simplicity, pass tokens in header (they're small ints)
    header["prompt_tokens"] = prompt_tokens
    return header, None


def build_start_session_result(
    session_id: int,
    semantic_token: int,
    hidden_state: torch.Tensor,
    eos: bool,
) -> tuple[dict, bytes]:
    """Build START_SESSION_RESULT message."""
    tensor_info = get_tensor_info(hidden_state)
    payload = tensor_to_bytes(hidden_state)

    header = {
        "op": Op.START_SESSION_RESULT,
        "session_id": session_id,
        "semantic_token": semantic_token,
        "eos": eos,
        "hidden_state_info": tensor_info,
    }
    return header, payload


def build_step(
    session_id: int,
    prev_codebooks: list[int],  # List of 11 token IDs (semantic + 10 codebooks)
    input_pos: int,
) -> tuple[dict, Optional[bytes]]:
    """Build STEP message."""
    header = {
        "op": Op.STEP,
        "session_id": session_id,
        "prev_codebooks": prev_codebooks,
        "input_pos": input_pos,
    }
    return header, None


def build_step_result(
    session_id: int,
    semantic_token: int,
    hidden_state: torch.Tensor,
    eos: bool,
    probs: Optional[torch.Tensor] = None,
) -> tuple[dict, Optional[bytes]]:
    """Build STEP_RESULT message."""
    tensor_info = get_tensor_info(hidden_state)
    payload = tensor_to_bytes(hidden_state)

    header = {
        "op": Op.STEP_RESULT,
        "session_id": session_id,
        "semantic_token": semantic_token,
        "eos": eos,
        "hidden_state_info": tensor_info,
    }

    if probs is not None:
        probs_info = get_tensor_info(probs)
        header["probs_info"] = probs_info
        payload += tensor_to_bytes(probs)

    return header, payload


def build_close_session(session_id: int) -> tuple[dict, None]:
    """Build CLOSE_SESSION message."""
    header = {
        "op": Op.CLOSE_SESSION,
        "session_id": session_id,
    }
    return header, None


def build_error(error_type: str, message: str) -> tuple[dict, None]:
    """Build ERROR message."""
    header = {
        "op": Op.ERROR,
        "error_type": error_type,
        "message": message,
    }
    return header, None


# ============================================================================
# Message Parsers
# ============================================================================


def parse_start_session(header: dict, payload: bytes) -> dict:
    """Parse START_SESSION message."""
    return {
        "session_id": header["session_id"],
        "prompt_tokens": header["prompt_tokens"],
        "sampling_config": header["sampling_config"],
        "input_seq_len": header["input_seq_len"],
    }


def parse_start_session_result(
    header: dict, payload: bytes
) -> SessionStartResult:
    """Parse START_SESSION_RESULT message."""
    hidden_state_info = header["hidden_state_info"]
    hidden_state = bytes_to_tensor(
        payload,
        dtype=dtype_from_string(hidden_state_info["dtype"]),
        shape=tuple(hidden_state_info["shape"]),
    )

    return SessionStartResult(
        semantic_token=header["semantic_token"],
        hidden_state=hidden_state,
        eos=header["eos"],
    )


def parse_step(header: dict, payload: bytes) -> dict:
    """Parse STEP message."""
    return {
        "session_id": header["session_id"],
        "prev_codebooks": header["prev_codebooks"],
        "input_pos": header["input_pos"],
    }


def parse_step_result(header: dict, payload: bytes) -> StepResult:
    """Parse STEP_RESULT message."""
    hidden_state_info = header["hidden_state_info"]

    # Extract hidden state from payload
    hidden_state_bytes = payload[: hidden_state_info["num_bytes"]]
    hidden_state = bytes_to_tensor(
        hidden_state_bytes,
        dtype=dtype_from_string(hidden_state_info["dtype"]),
        shape=tuple(hidden_state_info["shape"]),
    )

    probs = None
    if "probs_info" in header:
        probs_info = header["probs_info"]
        probs_bytes = payload[hidden_state_info["num_bytes"] :]
        probs = bytes_to_tensor(
            probs_bytes,
            dtype=dtype_from_string(probs_info["dtype"]),
            shape=tuple(probs_info["shape"]),
        )

    return StepResult(
        semantic_token=header["semantic_token"],
        hidden_state=hidden_state,
        eos=header["eos"],
        probs=probs,
    )


def parse_close_session(header: dict) -> int:
    """Parse CLOSE_SESSION message."""
    return header["session_id"]


# ============================================================================
# Socket Helpers
# ============================================================================


def recv_message(sock: socket.socket, timeout: float = 30.0) -> Optional[bytes]:
    """
    Receive a complete message from socket.

    Args:
        sock: Socket to receive from
        timeout: Socket timeout in seconds

    Returns:
        Message bytes, or None if connection closed
    """
    sock.settimeout(timeout)

    try:
        # Read header length
        header_len_data = _recv_exact(sock, 4)
        if not header_len_data:
            return None

        header_len = struct.unpack("<I", header_len_data)[0]

        # Read header
        header_bytes = _recv_exact(sock, header_len)
        if not header_bytes:
            return None

        # Read payload length
        payload_len_data = _recv_exact(sock, 8)
        if not payload_len_data:
            return None

        payload_len = struct.unpack("<Q", payload_len_data)[0]

        # Read payload
        payload = b""
        if payload_len > 0:
            payload = _recv_exact(sock, payload_len)
            if not payload:
                return None

        return header_len_data + header_bytes + payload_len_data + payload

    except socket.timeout:
        raise TimeoutError(f"Socket receive timeout after {timeout}s")
    except ConnectionError as e:
        raise ConnectionError(f"Socket connection error: {e}")


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Receive exactly n bytes from socket."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def send_message(sock: socket.socket, header: dict, payload: Optional[bytes] = None):
    """
    Send a message through socket.

    Args:
        sock: Socket to send through
        header: Message header dict
        payload: Optional binary payload
    """
    message = encode_message(header, payload)
    sock.sendall(message)
