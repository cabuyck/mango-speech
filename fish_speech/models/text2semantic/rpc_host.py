"""
Host-side components for RPC-based layer offload.
This module provides the HostPrefixRunner which owns embeddings and first N transformer layers.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    KVCache,
    precompute_freqs_cis,
)


class HostPrefixRunner(nn.Module):
    """
    Host-side runner that owns embeddings and first N transformer layers.
    This runs on the host machine and processes the prefix of the transformer.
    """

    def __init__(
        self,
        model: DualARTransformer,
        num_host_layers: int,
        device: str,
    ):
        """
        Initialize HostPrefixRunner from a full DualARTransformer model.

        Args:
            model: The full DualARTransformer model to split
            num_host_layers: Number of transformer layers to run on host
            device: Device to run on (e.g., "cuda")
        """
        super().__init__()

        self.config = model.config
        self.device = device
        self.num_host_layers = num_host_layers

        # Validate split point
        if num_host_layers < 0 or num_host_layers > self.config.n_layer:
            raise ValueError(
                f"Invalid num_host_layers={num_host_layers}, must be in [0, {self.config.n_layer}]"
            )

        logger.info(
            f"Creating HostPrefixRunner with {num_host_layers}/{self.config.n_layer} layers"
        )

        # Move embeddings from original model
        self.embeddings = model.embeddings
        self.codebook_embeddings = model.codebook_embeddings

        # Split transformer layers: host owns first N layers
        self.layers = nn.ModuleList(list(model.layers[:num_host_layers]))

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

        # Copy audio projector if present (for multi-modal inputs)
        if hasattr(model, "audio_projector"):
            self.audio_projector = model.audio_projector
        else:
            self.audio_projector = None

        # Cache configuration
        self.max_batch_size = -1
        self.max_seq_len = -1

        # For compatibility with inference code
        if hasattr(model, "tokenizer"):
            self.tokenizer = model.tokenizer

        # RPC connection info (set by launch_rpc_queue)
        self.vm_owner_name = None
        self.vm_address = None

        logger.info(f"HostPrefixRunner initialized with {len(self.layers)} layers")

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        """
        Setup KV caches for host layers only.
        VM layers will have their own caches on the VM worker.
        """
        if self.max_seq_len >= max_seq_len and self.max_batch_size >= max_batch_size:
            return

        from fish_speech.models.text2semantic.llama import find_multiple

        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        logger.info(
            f"Setting up host KV caches: max_batch_size={max_batch_size}, max_seq_len={max_seq_len}"
        )

        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.n_local_heads,
                self.config.head_dim,
                dtype=dtype,
            )

    def forward_generate(
        self,
        inp: Tensor,
        input_pos: Optional[Tensor] = None,
        audio_masks: Optional[Tensor] = None,
        audio_parts: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Run the prefix layers (embeddings + first N transformer layers).

        This method implements the same logic as BaseTransformer.forward_generate()
        but only runs the first N layers and returns intermediate hidden states.

        Args:
            inp: Input tokens (1, num_codebooks+1, seq_len)
            input_pos: Input positions
            audio_masks: Audio token masks
            audio_parts: Audio embedding parts

        Returns:
            Hidden states after processing through host layers (1, 1 or seq_len, dim)
        """
        # Replicate embedding logic from BaseTransformer.forward_generate()
        # (lines 400-421 in llama.py)

        # Codebook embeddings
        embeds = []
        for i in range(self.config.num_codebooks):
            emb = self.codebook_embeddings(
                inp[:, i + 1] + i * self.config.codebook_size
            )
            embeds.append(emb)

        vq_embeds_sum = torch.stack(embeds, dim=1).sum(dim=1)

        # Semantic token masking
        vq_masks = (inp[:, 0] >= self.config.semantic_begin_id) & (
            inp[:, 0] <= self.config.semantic_end_id
        )

        vq_embeds_sum[~vq_masks] = 0
        x = self.embeddings(inp[:, 0]) + vq_embeds_sum

        # Scale codebook embeddings if configured
        if self.config.scale_codebook_embeddings:
            vq_masks_expanded = vq_masks.unsqueeze(-1).expand_as(x)
            x = torch.where(
                vq_masks_expanded, x / math.sqrt(self.config.num_codebooks + 1), x
            )

        # Audio embeddings (if present)
        if audio_parts is not None and self.audio_projector is not None:
            audio_embeds = self.audio_projector(audio_parts)
            if self.config.scale_codebook_embeddings:
                x[audio_masks] = audio_embeds / math.sqrt(2)
            else:
                x[audio_masks] = audio_embeds

        # Position preparation (lines 435-442 in llama.py)
        if input_pos is None:
            input_pos = torch.arange(inp.shape[-1], device=x.device)
            max_seq_len = inp.shape[-1]
        else:
            max_seq_len = self.max_seq_len

        mask = self.causal_mask[None, None, input_pos, :max_seq_len]
        freqs_cis = self.freqs_cis[input_pos]

        # Run host layers (lines 444-445 in llama.py)
        for layer in self.layers:
            x = layer(x, freqs_cis, mask, input_pos=input_pos)

        # Return intermediate hidden state for VM to continue
        return x

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass compatibility method."""
        return self.forward_generate(inp)
