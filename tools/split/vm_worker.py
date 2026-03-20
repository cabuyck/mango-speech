#!/usr/bin/env python3
"""
VM Worker for Split Mode

This script runs on the VM with the 1080 Ti GPU and executes only the slow AR
(4B parameter LLaMA) component. It receives generation requests from the host
via TCP and returns semantic tokens + hidden states.

Usage:
    python tools/split/vm_worker.py \\
        --checkpoint-path checkpoints/s2-pro \\
        --device cuda:0 \\
        --port 50061
"""

import argparse
import socket
import threading
import traceback
from pathlib import Path

import torch
from loguru import logger

from fish_speech.models.text2semantic.llama import DualARTransformer
from fish_speech.models.text2semantic.split_protocol import (
    Op,
    SessionStartResult,
    StepResult,
    build_close_session,
    build_error,
    build_start_session_result,
    build_step_result,
    bytes_to_tensor,
    dtype_from_string,
    encode_message,
    parse_close_session,
    parse_start_session,
    parse_step,
    recv_message,
    send_message,
)


class VMWorker:
    """
    VM worker that runs slow AR inference for split mode.

    The worker:
    1. Loads only the slow AR components of the DualARTransformer
    2. Listens for TCP connections from the host
    3. Handles START_SESSION, STEP, and CLOSE_SESSION requests
    4. Returns semantic tokens and hidden states to the host
    """

    def __init__(self, checkpoint_path: str, device: str, port: int, dtype_str: str = "float16"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.port = port
        self.dtype = torch.float16 if dtype_str == "float16" else torch.float32

        self.model = None
        self._sessions = {}
        self._session_counter = 0
        self._lock = threading.Lock()

        # Pre-allocate sampling tensors
        self._temperature = None
        self._top_p = None

    def load_slow_only(self):
        """Load only slow AR components (token embeddings, slow layers, slow head)."""
        logger.info(f"Loading slow AR model from {self.checkpoint_path}")

        # Load full model first
        self.model = DualARTransformer.from_pretrained(
            self.checkpoint_path, load_weights=True
        )

        # Delete fast AR components to save memory
        logger.info("Deleting fast AR components to save VRAM")
        delattr(self.model, "fast_project_in")
        delattr(self.model, "fast_embeddings")
        delattr(self.model, "fast_layers")
        delattr(self.model, "fast_norm")
        delattr(self.model, "fast_output")
        delattr(self.model, "fast_freqs_cis")

        # Move to device with fp16 (1080 Ti doesn't support bf16)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        # Setup caches for slow layers only
        logger.info("Setting up KV caches for slow AR")
        self.model.setup_caches(
            max_batch_size=1,
            max_seq_len=self.model.config.max_seq_len,
            dtype=self.dtype,
        )

        # Pre-allocate sampling tensors
        self._temperature = torch.tensor(0.7, device=self.device, dtype=self.dtype)
        self._top_p = torch.tensor(0.7, device=self.device, dtype=self.dtype)

        logger.info(f"Slow AR model loaded on {self.device} with dtype {self.dtype}")

    def handle_client(self, conn: socket.socket, addr: tuple):
        """Handle one client connection."""
        logger.info(f"Connection from {addr}")

        try:
            while True:
                # Receive message
                data = recv_message(conn, timeout=300.0)  # 5 minute timeout
                if data is None:
                    logger.info(f"Client {addr} disconnected")
                    break

                from fish_speech.models.text2semantic.split_protocol import decode_message

                header, payload = decode_message(data)
                op = header["op"]

                if op == Op.START_SESSION:
                    result = self._handle_start_session(header, payload)
                    send_message(conn, result[0], result[1])
                elif op == Op.STEP:
                    result = self._handle_step(header, payload)
                    send_message(conn, result[0], result[1])
                elif op == Op.CLOSE_SESSION:
                    session_id = parse_close_session(header)
                    self._close_session(session_id)
                else:
                    logger.warning(f"Unknown opcode: {op}")
                    error_header, _ = build_error("unknown_opcode", f"Unknown opcode: {op}")
                    send_message(conn, error_header)

        except TimeoutError:
            logger.warning(f"Client {addr} timed out")
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}")
            logger.error(traceback.format_exc())
            try:
                error_header, _ = build_error(type(e).__name__, str(e))
                send_message(conn, error_header)
            except:
                pass
        finally:
            conn.close()

    def _handle_start_session(self, header: dict, payload: bytes) -> tuple:
        """Handle START_SESSION request."""
        params = parse_start_session(header, payload)
        session_id = params["session_id"]

        logger.debug(f"Starting session {session_id}")

        with self._lock:
            # Reconstruct prompt tensor
            num_codebooks = self.model.config.num_codebooks + 1
            seq_len = params["input_seq_len"]

            # Reshape flattened tokens back to (seq_len, num_codebooks)
            prompt_tokens_list = params["prompt_tokens"]
            prompt_tensor = torch.tensor(
                prompt_tokens_list,
                device=self.device,
                dtype=torch.long,
            ).reshape(seq_len, num_codebooks)

            sampling_config = params["sampling_config"]

            # Run slow AR forward pass on prompt
            input_pos = torch.arange(
                0, seq_len, device=self.device, dtype=torch.long
            )

            forward_result = self.model.forward_generate(
                prompt_tensor,
                input_pos,
                audio_masks=None,
                audio_parts=None,
            )

            logits = forward_result.logits
            hidden_state = forward_result.hidden_states

            # Build semantic logit bias
            vocab_size = self.model.config.vocab_size
            semantic_logit_bias = torch.full(
                (1, 1, vocab_size),
                float("-inf"),
                device=self.device,
                dtype=logits.dtype,
            )
            semantic_logit_bias[
                0,
                0,
                self.model.config.semantic_begin_id : self.model.config.semantic_end_id + 1,
            ] = 0.0

            from fish_speech.tokenizer import IM_END_TOKEN

            im_end_id = self.model.tokenizer.get_token_id(IM_END_TOKEN)
            semantic_logit_bias[0, 0, im_end_id] = 0.0

            # Sample semantic token
            from fish_speech.models.text2semantic.inference import sample

            biased_logits = logits + semantic_logit_bias

            semantic_token = sample(
                biased_logits,
                temperature=sampling_config["temperature"],
                top_p=sampling_config["top_p"],
                top_k=sampling_config["top_k"],
            )[0]

            eos = semantic_token.item() == im_end_id

            # Initialize session state (RAS window)
            self._sessions[session_id] = {
                "previous_tokens": torch.zeros(
                    (num_codebooks, 10),
                    dtype=torch.int,
                    device=self.device,
                ),
                "sampling_config": sampling_config,
            }

            # Build response
            result = build_start_session_result(
                session_id=session_id,
                semantic_token=semantic_token.item(),
                hidden_state=hidden_state[:, -1:, :].clone(),  # Last token only
                eos=eos,
            )

            logger.debug(f"Session {session_id} started, first token: {semantic_token.item()}")

            return result

    def _handle_step(self, header: dict, payload: bytes) -> tuple:
        """Handle STEP request."""
        params = parse_step(header, payload)
        session_id = params["session_id"]

        if session_id not in self._sessions:
            error_header, _ = build_error("invalid_session", f"Session {session_id} not found")
            return error_header, None

        session = self._sessions[session_id]

        with self._lock:
            # Reconstruct prev_codebooks tensor
            prev_codebooks_list = params["prev_codebooks"]
            num_codebooks = self.model.config.num_codebooks + 1

            # Shape: (1, num_codebooks, 1)
            prev_codebooks = torch.tensor(
                prev_codebooks_list,
                device=self.device,
                dtype=torch.long,
            ).reshape(1, num_codebooks, 1)

            input_pos = params["input_pos"]

            # Run slow AR forward pass
            forward_result = self.model.forward_generate(
                prev_codebooks,
                torch.tensor([input_pos], device=self.device, dtype=torch.long),
                audio_masks=None,
                audio_parts=None,
            )

            logits = forward_result.logits
            hidden_state = forward_result.hidden_states

            # Build semantic logit bias
            vocab_size = self.model.config.vocab_size
            semantic_logit_bias = torch.full(
                (1, 1, vocab_size),
                float("-inf"),
                device=self.device,
                dtype=logits.dtype,
            )
            semantic_logit_bias[
                0,
                0,
                self.model.config.semantic_begin_id : self.model.config.semantic_end_id + 1,
            ] = 0.0

            from fish_speech.tokenizer import IM_END_TOKEN

            im_end_id = self.model.tokenizer.get_token_id(IM_END_TOKEN)
            semantic_logit_bias[0, 0, im_end_id] = 0.0

            biased_logits = logits + semantic_logit_bias

            # RAS sampling
            previous_tokens = session["previous_tokens"]
            sampling_config = session["sampling_config"]

            from fish_speech.models.text2semantic.inference import (
                RAS_HIGH_TEMP,
                RAS_HIGH_TOP_P,
                sample,
            )

            main_token_normal = sample(
                biased_logits,
                temperature=sampling_config["temperature"],
                top_p=sampling_config["top_p"],
                top_k=sampling_config["top_k"],
            )[0]

            # High-temp sample for RAS
            high_temp = torch.tensor(
                RAS_HIGH_TEMP, device=self.device, dtype=logits.dtype
            )
            high_top_p = torch.tensor(RAS_HIGH_TOP_P, device=self.device, dtype=logits.dtype)

            main_token_high = sample(
                biased_logits, temperature=high_temp, top_p=high_top_p, top_k=30
            )[0]

            # Apply RAS logic
            in_window = (previous_tokens[0] == main_token_normal).any()
            is_semantic = (main_token_normal >= self.model.config.semantic_begin_id) & (
                main_token_normal <= self.model.config.semantic_end_id
            )
            should_use_high = in_window & is_semantic
            semantic_token = torch.where(
                should_use_high, main_token_high, main_token_normal
            )

            eos = semantic_token.item() == im_end_id

            # Update RAS window
            session["previous_tokens"] = session["previous_tokens"].roll(-1, dims=1)
            session["previous_tokens"][:, -1] = semantic_token

            # Build response
            result = build_step_result(
                session_id=session_id,
                semantic_token=semantic_token.item(),
                hidden_state=hidden_state.clone(),
                eos=eos,
                # probs=None  # Don't send probs to save bandwidth
            )

            return result

    def _close_session(self, session_id: int):
        """Close session and free resources."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.debug(f"Session {session_id} closed")


def main():
    parser = argparse.ArgumentParser(
        description="VM worker for split mode (runs slow AR on remote GPU)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (default: cuda:0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50061,
        help="Port to listen on (default: 50061)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Data type (default: float16 for 1080 Ti compatibility)",
    )

    args = parser.parse_args()

    # Create worker
    worker = VMWorker(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        port=args.port,
        dtype_str=args.dtype,
    )

    # Load model
    worker.load_slow_only()

    # Start server
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(5)

    logger.info(f"VM worker listening on {args.host}:{args.port}")
    logger.info("Ready to accept connections from host")

    try:
        while True:
            conn, addr = server_sock.accept()
            # Handle each client in a thread
            threading.Thread(
                target=worker.handle_client, args=(conn, addr), daemon=True
            ).start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()
