"""
VM-side components for RPC-based layer offload.
This module provides the VMSuffixRunner which owns remaining layers, norm, output, and fast AR.
"""

import math
from typing import Optional

import torch
import torch.distributed.rpc as rpc
from loguru import logger
from torch import Tensor
from torch.nn import functional as F

from fish_speech.models.text2semantic.inference_rpc import (
    RPCDecodePayload,
    RPCDecodeResponse,
    RAS_HIGH_TEMP,
    RAS_HIGH_TOP_P,
    RAS_WIN_SIZE,
    logits_to_probs,
    multinomial_sample_one_no_sync,
    sample,
)
from fish_speech.models.text2semantic.llama import (
    DualARTransformer,
    KVCache,
    precompute_freqs_cis,
)


class VMSuffixRunner(nn.Module):
    """
    VM-side runner that owns remaining transformer layers, norm, output, and fast AR.
    This runs on the VM worker and completes the slow AR, handles sampling, and runs fast AR.
    """

    def __init__(
        self,
        model: DualARTransformer,
        num_host_layers: int,
        device: str,
    ):
        """
        Initialize VMSuffixRunner from a full DualARTransformer model.

        Args:
            model: The full DualARTransformer model to split
            num_host_layers: Number of transformer layers on host (remaining layers go to VM)
            device: Device to run on (e.g., "cuda")
        """
        super().__init__()

        self.config = model.config
        self.device = device
        self.num_host_layers = num_host_layers
        self.num_vm_layers = model.config.n_layer - num_host_layers

        logger.info(
            f"Creating VMSuffixRunner with {self.num_vm_layers}/{model.config.n_layer} layers"
        )

        # Validate split point
        if num_host_layers < 0 or num_host_layers > model.config.n_layer:
            raise ValueError(
                f"Invalid num_host_layers={num_host_layers}, must be in [0, {model.config.n_layer}]"
            )

        # Split transformer layers: VM owns remaining layers
        self.layers = model.layers[num_host_layers:]

        # Copy norm and output/projection layers
        self.norm = model.norm

        # For tied embeddings: copy weight reference to VM
        # Memory cost: ~40MB for 32000 x 4096 embeddings
        if model.config.tie_word_embeddings:
            self.embeddings_weight = model.embeddings.weight
            logger.info("Using tied embeddings (copied weight reference to VM)")
        else:
            self.output = model.output
            self.embeddings_weight = None

        # Fast AR components (entire fast AR stack runs on VM)
        if hasattr(model, "fast_project_in"):
            self.fast_project_in = model.fast_project_in
        else:
            # If not present, create identity (shouldn't happen for DualAR)
            self.fast_project_in = torch.nn.Identity()

        self.fast_embeddings = model.fast_embeddings
        self.fast_layers = model.fast_layers
        self.fast_norm = model.fast_norm
        self.fast_output = model.fast_output

        # Copy position embeddings
        self.register_buffer(
            "freqs_cis",
            model.freqs_cis.clone(),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            model.causal_mask.clone(),
            persistent=False,
        )

        # Copy fast position embeddings
        self.register_buffer(
            "fast_freqs_cis",
            model.fast_freqs_cis.clone(),
            persistent=False,
        )

        # Cache configuration
        self.max_batch_size = -1
        self.max_seq_len = -1

        # For compatibility
        if hasattr(model, "tokenizer"):
            self.tokenizer = model.tokenizer

        logger.info(
            f"VMSuffixRunner initialized: {len(self.layers)} slow layers + "
            f"{len(self.fast_layers)} fast layers"
        )

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        """
        Setup KV caches for VM slow layers and fast layers.
        Host layers have their own caches on the host machine.
        """
        if self.max_seq_len >= max_seq_len and self.max_batch_size >= max_batch_size:
            return

        from fish_speech.models.text2semantic.llama import find_multiple

        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        logger.info(
            f"Setting up VM KV caches: max_batch_size={max_batch_size}, max_seq_len={max_seq_len}"
        )

        # Setup slow AR suffix layer caches
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.n_local_heads,
                self.config.head_dim,
                dtype=dtype,
            )

        # Setup fast AR caches (num_codebooks positions only)
        for layer in self.fast_layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size,
                self.config.num_codebooks,
                self.config.fast_n_local_heads,
                self.config.fast_head_dim,
                dtype=dtype,
            )

    def forward_generate(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Complete the slow AR by running remaining transformer layers.

        Args:
            x: Hidden states from host (1, 1 or seq_len, dim)
            input_pos: Input positions

        Returns:
            Logits (1, 1 or seq_len, vocab_size)
        """
        max_seq_len = self.max_seq_len

        # Prepare mask and freqs_cis
        mask = self.causal_mask[None, None, input_pos, :max_seq_len]
        freqs_cis = self.freqs_cis[input_pos]

        # Continue transformer layer loop (lines 444-445 in llama.py)
        for layer in self.layers:
            x = layer(x, freqs_cis, mask, input_pos=input_pos)

        # Only take last token if not in prefill mode
        if x.size(1) > 1:
            x = x[:, -1:]

        # Norm and logits (lines 450-457 in llama.py)
        slow_out = self.norm(x)

        if self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings_weight)
        else:
            token_logits = self.output(slow_out)

        return token_logits

    def forward_generate_fast(
        self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> Tensor:
        """
        Run fast AR generation for remaining codebooks.

        Args:
            x: Projected hidden states (1, 1, fast_dim)
            input_pos: Codebook position (0 to num_codebooks-1)

        Returns:
            Codebook logits (1, 1, codebook_size)
        """
        # Reshape for fast transformer
        x = x.view(x.shape[0], 1, -1)

        fast_mask = self.causal_mask[
            None, None, input_pos, : self.config.num_codebooks
        ]
        fast_freqs_cis = self.fast_freqs_cis[input_pos]

        for layer in self.fast_layers:
            x = layer(x, fast_freqs_cis, fast_mask, input_pos=input_pos)

        fast_out = self.fast_norm(x)
        codebook_logits = self.fast_output(fast_out)

        return codebook_logits

    def _sample_semantic_token(
        self,
        logits: Tensor,
        semantic_logit_bias: Tensor,
        temperature: Tensor,
        top_p: Tensor,
        top_k: int,
        previous_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample the main semantic token with Repetition Aware Sampling (RAS).

        This implements lines 118-144 from inference.py decode_one_token_ar().
        """
        # Apply constrained decoding: only allow semantic tokens + im_end
        biased_logits = logits + semantic_logit_bias

        # Normal sample
        main_token_normal = sample(
            biased_logits, temperature=temperature, top_p=top_p, top_k=top_k
        )[0]

        # RAS: also sample with high temp to use as fallback if token repeats
        high_temp = torch.tensor(
            RAS_HIGH_TEMP, device=temperature.device, dtype=temperature.dtype
        )
        high_top_p = torch.tensor(RAS_HIGH_TOP_P, device=top_p.device, dtype=top_p.dtype)
        main_token_high = sample(
            biased_logits, temperature=high_temp, top_p=high_top_p, top_k=top_k
        )[0]

        # Use high-temp sample if: token is semantic AND token is in previous window
        if previous_tokens is not None:
            in_window = (previous_tokens[0] == main_token_normal).any()
            is_semantic = (main_token_normal >= self.config.semantic_begin_id) & (
                main_token_normal <= self.config.semantic_end_id
            )
            should_use_high = in_window & is_semantic
            main_token_normal = torch.where(
                should_use_high, main_token_high, main_token_normal
            )

        return main_token_normal

    def _generate_fast_codebooks(
        self,
        hidden_states: Tensor,
        main_token: Tensor,
        temperature: Tensor,
        top_p: Tensor,
        top_k: int,
    ) -> Tensor:
        """
        Generate fast AR codebooks for the remaining codebook positions.

        This implements lines 146-176 from inference.py decode_one_token_ar().
        """
        codebooks = [main_token]

        # Reset input position for fast AR
        input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)

        # Project to fast dimension
        hidden_states = self.fast_project_in(hidden_states)

        # First codebook is from main semantic token
        a = codebooks[0] - self.config.semantic_begin_id
        a = torch.clamp(a, min=0, max=self.config.codebook_size - 1)

        hidden_states = self.fast_embeddings(a)
        codebooks.append(a)

        # Generate remaining codebooks
        for codebook_idx in range(1, self.config.num_codebooks):
            input_pos = torch.tensor(
                [codebook_idx], device=hidden_states.device, dtype=torch.long
            )
            logits = self.forward_generate_fast(hidden_states, input_pos)

            short_logits = logits  # DualAR predicts config.codebook_size number of tokens

            # Sample from codebook logits (no constrain for fast codebooks)
            a = sample(
                short_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )[0]

            hidden_states = self.fast_embeddings(a)
            codebooks.append(a)

        codebooks = torch.stack(codebooks, dim=1)

        return codebooks.T  # (num_codebooks+1, 1)

    @rpc.functions.async_execution
    async def run_decode_step(self, payload: RPCDecodePayload) -> RPCDecodeResponse:
        """
        RPC handler for a single decode step.
        Completes slow AR, samples semantic token, and runs fast AR.

        This is the main RPC entry point called by the host for each generated token.
        """
        try:
            # Step 1: Complete slow AR suffix layers
            logits = self.forward_generate(payload.x, payload.input_pos)
            hidden_states = payload.x  # Hidden states from host

            # Step 2: Sample semantic token
            main_token = self._sample_semantic_token(
                logits=logits,
                semantic_logit_bias=payload.semantic_logit_bias,
                temperature=payload.temperature,
                top_p=payload.top_p,
                top_k=payload.top_k,
                previous_tokens=payload.previous_tokens,
            )

            # Project hidden states for fast AR
            hidden_states_projected = self.fast_project_in(hidden_states)

            # Step 3: Fast AR codebook generation
            codebooks = self._generate_fast_codebooks(
                hidden_states=hidden_states_projected,
                main_token=main_token,
                temperature=payload.temperature,
                top_p=payload.top_p,
                top_k=payload.top_k,
            )

            return RPCDecodeResponse(codebooks=codebooks, success=True)

        except Exception as e:
            logger.error(f"VM decode step failed: {e}")
            import traceback

            return RPCDecodeResponse(
                codebooks=None, success=False, error_msg=traceback.format_exc()
            )
