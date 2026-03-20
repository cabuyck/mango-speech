"""
Backend Abstractions for Split Mode

Provides clean abstractions that allow swapping between local and remote
slow AR backends, enabling distributed inference across GPUs.
"""

import threading
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger

from fish_speech.models.text2semantic.llama import DualARTransformer
from fish_speech.models.text2semantic.split_protocol import (
    Op,
    SessionStartResult,
    StepResult,
    build_close_session,
    build_start_session,
    build_step,
    bytes_to_tensor,
    dtype_from_string,
    parse_start_session_result,
    parse_step_result,
    recv_message,
    send_message,
)


# ============================================================================
# Abstract Backend Interface
# ============================================================================


class SlowARBackend(ABC):
    """
    Abstract base for slow AR inference backends.

    The slow AR (LLaMA-based 4B parameter) model can run either:
    - Locally (LocalSlowARBackend): All components on same machine
    - Remotely (RemoteSlowARBackend): Slow AR on VM, fast AR locally
    """

    @abstractmethod
    def start_session(
        self,
        prompt_tensor: torch.Tensor,
        sampling_config: dict,
    ) -> SessionStartResult:
        """
        Initialize a new generation session with prompt tokens.

        Args:
            prompt_tensor: Tokenized prompt (seq_len, num_codebooks+1)
            sampling_config: Sampling parameters (temperature, top_p, top_k)

        Returns:
            SessionStartResult with initial semantic token and hidden state
        """
        pass

    @abstractmethod
    def step(
        self,
        session_id: int,
        prev_codebooks: torch.Tensor,
        input_pos: int,
    ) -> StepResult:
        """
        Generate one semantic token given previous codebooks.

        Args:
            session_id: Active session ID
            prev_codebooks: Previous codebook tokens (1, num_codebooks+1, 1)
            input_pos: Current position in sequence

        Returns:
            StepResult with semantic token, hidden state, EOS flag
        """
        pass

    @abstractmethod
    def close_session(self, session_id: int):
        """Clean up session resources."""
        pass


class FastARBackend:
    """
    Local fast AR backend (always runs locally, no remote support).

    The fast AR (400M parameter) model generates the 10 acoustic codebooks
    from the semantic token and hidden state produced by the slow AR.
    """

    def __init__(self, model: DualARTransformer, device: str):
        self.model = model
        self.device = device

    def decode_from_semantic(
        self,
        hidden_state: torch.Tensor,
        semantic_token: int,
        sampling_config: dict,
    ) -> torch.Tensor:
        """
        Generate all 10 codebook tokens from semantic token + hidden state.

        This extracts lines 148-176 from decode_one_token_ar().

        Args:
            hidden_state: Hidden state from slow AR (1, 1, hidden_dim)
            semantic_token: Semantic token ID
            sampling_config: Sampling params (temperature, top_p, top_k)

        Returns:
            Codebooks tensor (1, num_codebooks+1, 1) - all 11 codebooks
        """
        temperature = sampling_config["temperature"]
        top_p = sampling_config["top_p"]
        top_k = sampling_config["top_k"]

        codebooks = [semantic_token]

        # Start fast AR processing
        input_pos = torch.tensor([0], device=hidden_state.device, dtype=torch.long)

        # Get semantic token embedding for fast AR
        a = semantic_token - self.model.config.semantic_begin_id
        a = torch.clamp(a, min=0, max=self.model.config.codebook_size - 1)

        hidden_states = self.model.fast_embeddings(a)

        codebooks.append(a)

        # Generate remaining 9 codebooks
        for codebook_idx in range(1, self.model.config.num_codebooks):
            input_pos = torch.tensor(
                [codebook_idx], device=hidden_states.device, dtype=torch.long
            )
            logits = self.model.forward_generate_fast(hidden_states, input_pos)

            # Sample codebook token
            from fish_speech.models.text2semantic.inference import sample

            a = sample(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )[0]

            hidden_states = self.model.fast_embeddings(a)
            codebooks.append(a)

        codebooks = torch.stack(codebooks, dim=1)
        return codebooks.T


# ============================================================================
# Local Backend (for testing and non-split mode)
# ============================================================================


class LocalSlowARBackend(SlowARBackend):
    """
    Local slow AR backend - wraps existing model for testing and non-split mode.

    This allows the same code path to be used whether running in split mode
    or locally, making testing and debugging easier.
    """

    def __init__(self, model: DualARTransformer, device: str):
        self.model = model
        self.device = device
        self._sessions = {}
        self._session_counter = 0
        self._lock = threading.Lock()

        # RAS window state
        self._previous_tokens = None

    def start_session(
        self,
        prompt_tensor: torch.Tensor,
        sampling_config: dict,
    ) -> SessionStartResult:
        """Run slow AR on prompt and return first semantic token."""
        with self._lock:
            session_id = self._session_counter
            self._session_counter += 1

            # Run forward_generate on prompt
            audio_masks = None  # TODO: Support audio masks
            audio_parts = None

            input_pos = torch.arange(
                0, prompt_tensor.shape[1], device=self.device, dtype=torch.long
            )

            forward_result = self.model.forward_generate(
                prompt_tensor,
                input_pos,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
            )

            logits = forward_result.logits
            hidden_state = forward_result.hidden_states

            # Sample semantic token with constrained decoding
            from fish_speech.models.text2semantic.inference import sample

            # Build semantic logit bias
            vocab_size = self.model.config.vocab_size
            semantic_logit_bias = torch.full(
                (1, 1, vocab_size), float("-inf"), device=self.device, dtype=logits.dtype
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

            semantic_token = sample(
                biased_logits,
                temperature=sampling_config["temperature"],
                top_p=sampling_config["top_p"],
                top_k=sampling_config["top_k"],
            )[0]

            # Check for EOS
            eos = semantic_token.item() == im_end_id

            # Store session state (including sampling_config for later steps)
            self._sessions[session_id] = {
                "previous_tokens": torch.zeros(
                    (self.model.config.num_codebooks + 1, 10),
                    dtype=torch.int,
                    device=self.device,
                ),
                "sampling_config": sampling_config,
            }

            return SessionStartResult(
                semantic_token=semantic_token.item(),
                hidden_state=hidden_state[:, -1:, :].clone(),
                eos=eos,
            )

    def step(
        self,
        session_id: int,
        prev_codebooks: torch.Tensor,
        input_pos: int,
    ) -> StepResult:
        """Generate one semantic token using slow AR."""
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")

            session = self._sessions[session_id]
            sampling_config = session["sampling_config"]

            # Run forward_generate
            forward_result = self.model.forward_generate(
                prev_codebooks,
                torch.tensor([input_pos], device=self.device, dtype=torch.long),
                audio_masks=None,
                audio_parts=None,
            )

            logits = forward_result.logits
            hidden_state = forward_result.hidden_states

            # Sample semantic token
            from fish_speech.models.text2semantic.inference import sample

            vocab_size = self.model.config.vocab_size
            semantic_logit_bias = torch.full(
                (1, 1, vocab_size), float("-inf"), device=self.device, dtype=logits.dtype
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

            # Get previous token for RAS
            previous_tokens = session["previous_tokens"]

            main_token_normal = sample(
                biased_logits,
                temperature=sampling_config["temperature"],
                top_p=sampling_config["top_p"],
                top_k=sampling_config["top_k"],
            )[0]

            # RAS
            from fish_speech.models.text2semantic.inference import (
                RAS_HIGH_TEMP,
                RAS_HIGH_TOP_P,
            )

            high_temp = torch.tensor(
                RAS_HIGH_TEMP, device=self.device, dtype=logits.dtype
            )
            high_top_p = torch.tensor(RAS_HIGH_TOP_P, device=self.device, dtype=logits.dtype)

            main_token_high = sample(
                biased_logits, temperature=high_temp, top_p=high_top_p, top_k=30
            )[0]

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

            return StepResult(
                semantic_token=semantic_token.item(),
                hidden_state=hidden_state.clone(),
                eos=eos,
                probs=None,
            )

    def close_session(self, session_id: int):
        """Clean up session resources."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]


# ============================================================================
# Remote Backend (client-side for split mode)
# ============================================================================


class RemoteSlowARBackend(SlowARBackend):
    """
    Client-side remote backend that connects to VM worker.

    This backend runs on the host machine and communicates with the VM
    via TCP socket to execute slow AR inference on the remote GPU.
    """

    def __init__(self, addr: str, port: int, device: str):
        self.addr = addr
        self.port = port
        self.device = device
        self._socket = None
        self._session_id = 0
        self._lock = threading.Lock()

    def _connect(self):
        """Establish TCP connection to VM worker."""
        if self._socket is not None:
            return

        logger.info(f"Connecting to remote slow AR worker at {self.addr}:{self.port}")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self.addr, self.port))
        logger.info("Connected to remote slow AR worker")

    def _ensure_connected(self):
        """Ensure connection is established."""
        if self._socket is None:
            self._connect()

    def start_session(
        self,
        prompt_tensor: torch.Tensor,
        sampling_config: dict,
    ) -> SessionStartResult:
        """Start remote generation session."""
        with self._lock:
            self._ensure_connected()

            session_id = self._session_id
            self._session_id += 1

            # Flatten prompt tokens for transmission
            prompt_tokens = prompt_tensor.flatten().tolist()

            header, _ = build_start_session(
                session_id=session_id,
                prompt_tokens=prompt_tokens,
                sampling_config=sampling_config,
                input_seq_len=prompt_tensor.shape[1],
            )

            send_message(self._socket, header)

            # Receive response
            response_data = recv_message(self._socket)
            if response_data is None:
                raise ConnectionError("Connection closed by remote worker")

            from fish_speech.models.text2semantic.split_protocol import decode_message

            resp_header, resp_payload = decode_message(response_data)

            if resp_header["op"] == Op.ERROR:
                raise RuntimeError(
                    f"Remote error: {resp_header.get('error_type')}: {resp_header.get('message')}"
                )

            result = parse_start_session_result(resp_header, resp_payload)

            # Move tensor to correct device
            result.hidden_state = result.hidden_state.to(self.device)

            return result

    def step(
        self,
        session_id: int,
        prev_codebooks: torch.Tensor,
        input_pos: int,
    ) -> StepResult:
        """Execute remote generation step."""
        with self._lock:
            self._ensure_connected()

            # Flatten codebooks for transmission
            codebook_list = prev_codebooks.flatten().tolist()

            header, _ = build_step(
                session_id=session_id,
                prev_codebooks=codebook_list,
                input_pos=input_pos,
            )

            send_message(self._socket, header)

            # Receive response
            response_data = recv_message(self._socket)
            if response_data is None:
                raise ConnectionError("Connection closed by remote worker")

            from fish_speech.models.text2semantic.split_protocol import decode_message

            resp_header, resp_payload = decode_message(response_data)

            if resp_header["op"] == Op.ERROR:
                raise RuntimeError(
                    f"Remote error: {resp_header.get('error_type')}: {resp_header.get('message')}"
                )

            result = parse_step_result(resp_header, resp_payload)

            # Move tensors to correct device
            result.hidden_state = result.hidden_state.to(self.device)
            if result.probs is not None:
                result.probs = result.probs.to(self.device)

            return result

    def close_session(self, session_id: int):
        """Close remote session."""
        with self._lock:
            if self._socket is None:
                return

            header, _ = build_close_session(session_id)
            try:
                send_message(self._socket, header)
            except Exception as e:
                logger.warning(f"Error closing session {session_id}: {e}")


# Import socket at module level
import socket
