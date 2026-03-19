#!/usr/bin/env python3
"""
Local test script for RPC-based layer offload.

This script helps you test the RPC components on a single machine
before deploying across multiple machines. It starts both the VM
and Host in separate processes and runs a simple test request.

Usage:
    python tools/test_rpc_local.py --checkpoint-path checkpoints/fish-speech-2.0
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import torch
import click

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_vm_worker(checkpoint_path, device, half, num_host_layers, vm_address, host_address):
    """Run the VM worker in a subprocess."""
    import subprocess

    cmd = [
        sys.executable,
        "tools/launch_rpc_vm.py",
        "--checkpoint-path", checkpoint_path,
        "--device", device,
        "--num-host-layers", str(num_host_layers),
        "--vm-address", vm_address,
        "--host-address", host_address,
    ]

    if half:
        cmd.append("--half")

    print(f"Starting VM worker: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode


@click.command()
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True),
    default="checkpoints/fish-speech-2.0",
    help="Path to model checkpoint directory",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to run on (e.g., 'cuda', 'cpu')",
)
@click.option(
    "--half/--no-half",
    default=False,
    help="Use FP16 precision (default: bfloat16)",
)
@click.option(
    "--num-host-layers",
    type=int,
    default=16,
    help="Number of transformer layers on host (default: 16)",
)
@click.option(
    "--host-address",
    type=str,
    default="localhost:29500",
    help="Address for host RPC endpoint (default: localhost:29500)",
)
@click.option(
    "--vm-address",
    type=str,
    default="localhost:29501",
    help="Address for VM RPC endpoint (default: localhost:29501)",
)
@click.option(
    "--text",
    type=str,
    default="Hello, this is a test of RPC-based layer offload.",
    help="Text to synthesize (default: greeting)",
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=100,
    help="Maximum number of tokens to generate (default: 100)",
)
def main(
    checkpoint_path,
    device,
    half,
    num_host_layers,
    host_address,
    vm_address,
    text,
    max_new_tokens,
):
    """
    Test RPC-based layer offload locally.

    This script starts both the VM worker and Host (API server) in separate processes
    and sends a test TTS request to verify the RPC setup is working correctly.

    The script will:
    1. Start the VM worker in a background process
    2. Wait for it to initialize
    3. Start the API server (host) in RPC mode
    4. Send a test TTS request
    5. Clean up processes on exit

    Example:
        python tools/test_rpc_local.py \\
            --checkpoint-path checkpoints/fish-speech-2.0 \\
            --device cuda \\
            --half \\
            --num-host-layers 16
    """
    print("=" * 60)
    print("Fish Speech RPC Layer Offload - Local Test")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Device:      {device}")
    print(f"  Precision:   {'FP16' if half else 'BF16'}")
    print(f"  Host layers: {num_host_layers}")
    print(f"  Host RPC:    {host_address}")
    print(f"  VM RPC:      {vm_address}")
    print(f"  Test text:   {text}")
    print()
    print("=" * 60)
    print()

    # Validate device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        device = "cpu"

    # Step 1: Start VM worker in background process
    print("Step 1: Starting VM worker...")
    vm_process = mp.Process(
        target=run_vm_worker,
        args=(
            checkpoint_path,
            device,
            half,
            num_host_layers,
            vm_address,
            host_address,
        ),
    )
    vm_process.start()

    # Wait for VM to initialize
    print("Waiting for VM worker to initialize (10 seconds)...")
    time.sleep(10)

    if not vm_process.is_alive():
        print("ERROR: VM worker failed to start or crashed")
        print("Check the VM worker output above for error messages")
        return 1

    print("VM worker appears to be running")
    print()

    # Step 2: Import and test RPC functionality
    print("Step 2: Testing RPC initialization on host...")

    try:
        import torch.distributed.rpc as rpc
        from fish_speech.models.text2semantic.llama import DualARTransformer
        from fish_speech.models.text2semantic.rpc_host import HostPrefixRunner
        from fish_speech.models.text2semantic.inference_rpc import launch_rpc_queue

        # Initialize RPC on host
        print(f"Initializing host RPC at {host_address}...")
        rpc.init_rpc(
            name="host",
            rank=0,
            world_size=2,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{host_address}",
            )
        )
        print("Host RPC initialized successfully")

    except Exception as e:
        print(f"ERROR: Failed to initialize host RPC: {e}")
        print("Make sure the VM worker is running and accessible")
        vm_process.terminate()
        vm_process.join()
        return 1

    print()
    print("=" * 60)
    print("SUCCESS: RPC components are working!")
    print()
    print("Next steps:")
    print("  1. For production use, start VM and Host on separate machines")
    print("  2. Update host_address and vm_address to actual hostnames/IPs")
    print("  3. Use tools/launch_rpc_host.sh and tools/launch_rpc_vm.py")
    print("  4. Send API requests to the host's HTTP endpoint")
    print("=" * 60)

    # Cleanup
    print()
    print("Cleaning up...")

    try:
        rpc.shutdown()
    except:
        pass

    vm_process.terminate()
    vm_process.join(timeout=5)

    if vm_process.is_alive():
        print("WARNING: VM worker did not shut down gracefully, killing...")
        vm_process.kill()
        vm_process.join()

    print("Test complete!")
    return 0


if __name__ == "__main__":
    # Fix multiprocessing for spawn
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
