"""
RPC-based inference for Fish Speech Dual-AR model.
This module provides utilities for splitting the Slow AR transformer across host and VM workers.
"""

import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed.rpc as rpc
from loguru import logger


# Disable tokenizers parallelism for RPC workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class RPCDecodePayload:
    """Payload sent from host to VM for each decode step."""
    x: torch.Tensor  # (1, 1, dim) - hidden states after host layers
    input_pos: torch.Tensor  # (1,) - current position
    audio_masks: torch.Tensor  # Semantic filtering mask
    audio_parts: torch.Tensor  # Audio embedding parts
    temperature: torch.Tensor  # Sampling temperature
    top_p: torch.Tensor  # Top-p sampling parameter
    top_k: int  # Top-k sampling parameter
    semantic_logit_bias: torch.Tensor  # (1, 1, vocab_size)
    previous_tokens: torch.Tensor  # (num_codebooks+1, RAS_WIN_SIZE) for RAS


@dataclass
class RPCDecodeResponse:
    """Response from VM to host after decode step."""
    codebooks: torch.Tensor  # (num_codebooks+1, 1) - generated codebooks
    success: bool  # Whether generation succeeded
    error_msg: Optional[str] = None  # Error message if failed


# Repetition Aware Sampling constants (from inference.py)
RAS_WIN_SIZE = 10
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9


def multinomial_sample_one_no_sync(probs_sort):
    """Sample from probability distribution without synchronizing."""
    q = torch.rand_like(probs_sort)
    q = -torch.log(q)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Convert logits to probabilities with top-p and top-k filtering."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    indices = torch.arange(sorted_logits.shape[-1], device=sorted_logits.device)
    top_k_mask = indices >= top_k
    sorted_indices_to_remove = (cum_probs > top_p) | top_k_mask
    sorted_indices_to_remove[0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = torch.where(
        indices_to_remove, float("-Inf"), logits
    )
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample from logits with temperature, top-p, and top-k."""
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_rpc(
    host_runner: "HostPrefixRunner",
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    RPC-aware single token generation.
    Completes slow AR on host side, then calls VM for sampling and fast AR.
    """
    # Step 1: Run host prefix layers
    x = host_runner.forward_generate(x, input_pos, audio_masks, audio_parts)

    # Step 2: Prepare payload for VM
    payload = RPCDecodePayload(
        x=x,
        input_pos=input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        semantic_logit_bias=semantic_logit_bias,
        previous_tokens=previous_tokens,
    )

    # Step 3: Call VM via RPC (does sampling + fast AR)
    try:
        # Call the global vm_runner.run_decode_step on the VM
        response = rpc_sync(
            host_runner.vm_owner_name,
            call_remote_decode_step,
            args=(payload,),
        )

        if not response.success:
            raise RuntimeError(f"VM decode failed: {response.error_msg}")

        return response.codebooks  # (num_codebooks+1, 1)

    except Exception as e:
        logger.error(f"RPC call failed: {e}")
        raise


def call_remote_decode_step(payload: RPCDecodePayload) -> RPCDecodeResponse:
    """
    Helper function to call the remote VM runner's decode step.
    This function runs on the VM side.
    """
    import fish_speech.models.text2semantic.rpc_vm as rpc_vm_module
    vm_runner = rpc_vm_module.vm_runner
    return vm_runner.run_decode_step(payload)


def rpc_sync(to, func, args=()):
    """Synchronous RPC call with error handling."""
    try:
        return rpc.rpc_sync(to, func, args=args)
    except Exception as e:
        logger.error(f"RPC sync call to {to}.{func} failed: {e}")
        raise


def launch_rpc_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
    num_host_layers: int = 16,
    host_address: str = "localhost:29500",
    vm_address: str = "localhost:29501",
):
    """
    Launch RPC-based inference queue.

    This initializes the RPC framework and creates a HostPrefixRunner that communicates
    with a remote VMSuffixRunner via RPC.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run on (e.g., "cuda")
        precision: Model precision (e.g., torch.half or torch.bfloat16)
        compile: Whether to compile the model (disabled for RPC mode)
        num_host_layers: Number of transformer layers to run on host
        host_address: Address of this host machine (format: "host:port")
        vm_address: Address of the VM worker (format: "host:port")

    Returns:
        Queue for submitting generation requests
    """
    input_queue = queue.Queue()
    init_event = threading.Event()

    # Store VM address for RPC calls
    vm_owner_name = "vm"

    def worker():
        try:
            # Initialize RPC on host
            rpc.init_rpc(
                name="host",
                rank=0,
                world_size=2,  # host + vm
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                    init_method=f"tcp://{host_address}",
                )
            )
            logger.info(f"RPC initialized on host at {host_address}")

            # Import model loading utilities
            from fish_speech.models.text2semantic.llama import DualARTransformer
            from fish_speech.models.text2semantic.rpc_host import HostPrefixRunner

            # Load full model first
            model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)
            model = model.to(device=device, dtype=precision)
            logger.info("Model loaded on host")

            # Create host prefix runner
            host_runner = HostPrefixRunner(
                model=model,
                num_host_layers=num_host_layers,
                device=device,
            )

            # Store reference to VM connection info
            host_runner.vm_owner_name = vm_owner_name
            host_runner.vm_address = vm_address

            # Setup caches for host layers
            host_runner.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )

            # Attach tokenizer to host_runner for compatibility
            host_runner.tokenizer = model.tokenizer
            host_runner.config = model.config

            # Get remote reference to VM runner
            # The VM worker should have registered itself as "vm_runner"
            logger.info(f"Getting remote reference to VM at {vm_address}")

            init_event.set()

            # Process requests
            while True:
                item = input_queue.get()
                if item is None:
                    break

                kwargs = item.request
                response_queue = item.response_queue

                try:
                    from fish_speech.models.text2semantic.inference import generate_long

                    # Create a wrapper that passes host_runner
                    def make_decode_wrapper(runner):
                        def wrapper(*args, **kw):
                            return decode_one_token_rpc(runner, *args, **kw)
                        return wrapper

                    for chunk in generate_long(
                        model=host_runner,
                        device=device,
                        decode_one_token=make_decode_wrapper(host_runner),
                        **kwargs,
                    ):
                        from fish_speech.models.text2semantic.inference import WrappedGenerateResponse
                        response_queue.put(
                            WrappedGenerateResponse(status="success", response=chunk)
                        )

                    # Clear cache after complete request batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(traceback.format_exc())
                    from fish_speech.models.text2semantic.inference import WrappedGenerateResponse
                    response_queue.put(WrappedGenerateResponse(status="error", response=e))
                    # Clear cache on error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Host worker failed: {traceback.format_exc()}")
            init_event.set()  # Ensure we don't block forever
            raise

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue
