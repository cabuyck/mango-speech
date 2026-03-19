#!/usr/bin/env python3
"""
VM worker launcher for RPC-based layer offload.
This script runs on the VM worker and initializes the VMSuffixRunner.
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed.rpc as rpc
import click
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fish_speech.models.text2semantic.llama import DualARTransformer
from fish_speech.models.text2semantic.rpc_vm import VMSuffixRunner

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@click.command()
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True),
    required=True,
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
    help="Number of transformer layers on host (remaining layers run on VM)",
)
@click.option(
    "--vm-address",
    type=str,
    default="localhost:29501",
    help="Address of this VM worker (format: 'host:port')",
)
@click.option(
    "--host-address",
    type=str,
    default="localhost:29500",
    help="Address of the host machine (format: 'host:port')",
)
@click.option(
    "--max-seq-len",
    type=int,
    default=8192,
    help="Maximum sequence length for KV cache",
)
def main(
    checkpoint_path: str,
    device: str,
    half: bool,
    num_host_layers: int,
    vm_address: str,
    host_address: str,
    max_seq_len: int,
):
    """
    Launch the VM worker for RPC-based layer offload.

    This script initializes the VMSuffixRunner and sets up the RPC server
    to receive decode requests from the host machine.

    Example:
        python tools/launch_rpc_vm.py \\
            --checkpoint-path checkpoints/fish-speech-2.0 \\
            --device cuda \\
            --half \\
            --num-host-layers 16 \\
            --vm-address localhost:29501 \\
            --host-address localhost:29500
    """
    logger.info("Starting VM worker for RPC-based layer offload")

    # Validate device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    # Determine precision
    precision = torch.half if half else torch.bfloat16
    logger.info(f"Using precision: {precision}")

    # Parse addresses
    vm_name, vm_port = vm_address.split(":")
    vm_port = int(vm_port)

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    t0 = torch.time()

    model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)
    model = model.to(device=device, dtype=precision)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Model loaded in {torch.time() - t0:.2f} seconds")

    # Create VM suffix runner
    logger.info(f"Creating VMSuffixRunner with {model.config.n_layer - num_host_layers} layers")
    vm_runner = VMSuffixRunner(
        model=model,
        num_host_layers=num_host_layers,
        device=device,
    )

    # Setup KV caches for VM layers
    vm_runner.setup_caches(
        max_batch_size=1,
        max_seq_len=max_seq_len,
        dtype=next(model.parameters()).dtype,
    )
    logger.info(f"VM KV caches initialized with max_seq_len={max_seq_len}")

    # Initialize RPC
    logger.info(f"Initializing RPC at {vm_address}, connecting to host at {host_address}")
    rpc.init_rpc(
        name="vm",
        rank=1,
        world_size=2,  # host + vm
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method=f"tcp://{host_address}",
        )
    )
    logger.info("RPC initialized successfully")

    # Register VM runner globally for RPC access
    # This allows the host to call vm_runner.run_decode_step()
    import fish_speech.models.text2semantic.rpc_vm as rpc_vm_module
    rpc_vm_module.vm_runner = vm_runner
    logger.info("VMSuffixRunner registered for RPC access")

    # Print VM configuration
    logger.info("=" * 50)
    logger.info("VM Worker Configuration:")
    logger.info(f"  Model: {checkpoint_path}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Precision: {precision}")
    logger.info(f"  Slow AR layers: {len(vm_runner.layers)} (layers {num_host_layers}-{model.config.n_layer-1})")
    logger.info(f"  Fast AR layers: {len(vm_runner.fast_layers)}")
    logger.info(f"  Max sequence length: {max_seq_len}")
    logger.info(f"  VM address: {vm_address}")
    logger.info(f"  Host address: {host_address}")
    logger.info("=" * 50)

    # Keep the worker alive
    logger.info("VM worker ready and waiting for RPC requests...")
    logger.info("Press Ctrl+C to exit")

    try:
        # Block forever to keep the RPC server alive
        while True:
            import time

            time.sleep(3600)  # Sleep in 1-hour increments
    except KeyboardInterrupt:
        logger.info("Shutting down VM worker")
        rpc.shutdown()
        logger.info("RPC shutdown complete")


if __name__ == "__main__":
    main()
