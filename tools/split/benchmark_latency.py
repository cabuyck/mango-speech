#!/usr/bin/env python3
"""
TCP Latency Benchmark for Split Mode

Measures round-trip latency between host and VM for typical tensor payloads.
This helps verify that the network is fast enough before implementing full split mode.

Acceptance criteria: Median round-trip < 1ms for local network between host and VM.
"""

import argparse
import socket
import struct
import time
from typing import Tuple

import numpy as np
import torch


def encode_message(header: dict, payload: bytes | None = None) -> bytes:
    """Encode message with header length prefix and payload length prefix."""
    import msgpack

    header_bytes = msgpack.packb(header, use_bin_type=True)

    header_len = struct.pack("<I", len(header_bytes))
    payload_len = struct.pack("<Q", len(payload) if payload else 0)

    message = header_len + header_bytes + payload_len
    if payload:
        message += payload

    return message


def decode_message(data: bytes) -> Tuple[dict, bytes]:
    """Decode message with header and payload."""
    import msgpack

    header_len = struct.unpack("<I", data[:4])[0]
    header = msgpack.unpackb(data[4 : 4 + header_len], raw=False)

    payload_len = struct.unpack("<Q", data[4 + header_len : 4 + header_len + 8])[0]
    payload = data[4 + header_len + 8 : 4 + header_len + 8 + payload_len] if payload_len else b""

    return header, payload


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert tensor to bytes for network transmission."""
    return memoryview(tensor.contiguous()).tobytes()


def bytes_to_tensor(data: bytes, dtype: str, shape: Tuple[int, ...]) -> torch.Tensor:
    """Reconstruct tensor from bytes."""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    tensor = torch.frombuffer(np.frombuffer(data, dtype=dtype), dtype=dtype_map[dtype])
    return tensor.reshape(shape)


def run_server(host: str = "0.0.0.0", port: int = 50061, tensor_size: int = 4096):
    """Run latency benchmark server."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)

    print(f"Latency benchmark server listening on {host}:{port}")

    while True:
        conn, addr = server_sock.accept()
        print(f"Connection from {addr}")

        try:
            while True:
                # Receive header length
                header_len_data = conn.recv(4)
                if not header_len_data:
                    break

                header_len = struct.unpack("<I", header_len_data)[0]
                header_bytes = conn.recv(header_len)

                payload_len_data = conn.recv(8)
                payload_len = struct.unpack("<Q", payload_len_data)[0]

                payload = b""
                if payload_len > 0:
                    remaining = payload_len
                    while remaining > 0:
                        chunk = conn.recv(min(remaining, 65536))
                        if not chunk:
                            break
                        payload += chunk
                        remaining -= len(chunk)

                header, payload_data = decode_message(
                    header_len_data + header_bytes + payload_len_data + payload
                )

                if header["op"] == "ping":
                    # Echo back the payload
                    response = encode_message({"op": "pong"}, payload_data)
                    conn.sendall(response)
                elif header["op"] == "shutdown":
                    break

        finally:
            conn.close()


def run_client(
    server_addr: str, server_port: int, iterations: int = 1000, tensor_size: int = 4096
):
    """Run latency benchmark client."""
    # Create test tensor (typical hidden state size)
    tensor = torch.randn(1, 1, tensor_size, dtype=torch.float16)
    payload = tensor_to_bytes(tensor)

    latencies = []

    print(f"Connecting to {server_addr}:{server_port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_addr, server_port))

    print(f"Running {iterations} iterations...")

    for i in range(iterations):
        start = time.perf_counter()

        # Send ping with tensor payload
        request = encode_message({"op": "ping", "iter": i}, payload)
        sock.sendall(request)

        # Receive response
        header_len_data = sock.recv(4)
        header_len = struct.unpack("<I", header_len_data)[0]
        header_bytes = sock.recv(header_len)

        payload_len_data = sock.recv(8)
        payload_len = struct.unpack("<Q", payload_len_data)[0]

        response_payload = b""
        if payload_len > 0:
            remaining = payload_len
            while remaining > 0:
                chunk = sock.recv(min(remaining, 65536))
                if not chunk:
                    break
                response_payload += chunk
                remaining -= len(chunk)

        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{iterations} iterations")

    # Send shutdown
    request = encode_message({"op": "shutdown"}, None)
    sock.sendall(request)

    sock.close()

    # Calculate statistics
    latencies_array = np.array(latencies)
    min_latency = np.min(latencies_array)
    max_latency = np.max(latencies_array)
    mean_latency = np.mean(latencies_array)
    median_latency = np.median(latencies_array)
    p95_latency = np.percentile(latencies_array, 95)
    p99_latency = np.percentile(latencies_array, 99)

    print("\n" + "=" * 60)
    print("Latency Benchmark Results")
    print("=" * 60)
    print(f"Iterations:       {iterations}")
    print(f"Tensor size:      {tensor_size} elements ({len(payload)} bytes)")
    print(f"Min latency:      {min_latency:.4f} ms")
    print(f"Max latency:      {max_latency:.4f} ms")
    print(f"Mean latency:     {mean_latency:.4f} ms")
    print(f"Median latency:   {median_latency:.4f} ms")
    print(f"95th percentile:  {p95_latency:.4f} ms")
    print(f"99th percentile:  {p99_latency:.4f} ms")
    print("=" * 60)

    if median_latency < 1.0:
        print("✓ PASS: Median latency < 1ms")
    else:
        print("✗ FAIL: Median latency >= 1ms (network may be too slow)")

    return median_latency < 1.0


def main():
    parser = argparse.ArgumentParser(
        description="TCP latency benchmark for split mode"
    )
    parser.add_argument(
        "--mode",
        choices=["server", "client"],
        required=True,
        help="Run as server or client",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0 for server, 127.0.0.1 for client)",
    )
    parser.add_argument("--port", type=int, default=50061, help="Server port")
    parser.add_argument(
        "--server-addr",
        type=str,
        default="127.0.0.1",
        help="Server address for client mode",
    )
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of iterations (client)"
    )
    parser.add_argument(
        "--tensor-size",
        type=int,
        default=4096,
        help="Tensor size in elements (client, default: 4096 for typical hidden state)",
    )

    args = parser.parse_args()

    if args.mode == "server":
        host = args.host if args.host != "0.0.0.0" else "0.0.0.0"
        run_server(host, args.port)
    else:
        run_client(args.server_addr, args.port, args.iterations, args.tensor_size)


if __name__ == "__main__":
    main()
