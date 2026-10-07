"""One-layer CPU inference using Transformers 4.57.6 native decoder operations.

Only local safetensors checkpoints are read. ``bf16`` selects unchanged native
SDPA attention (the constructor's dtype may be float32 for reference tests).
Other variants replace only its post-RoPE attention call, not the decoder forward.
Inputs are equal-length, unpadded sequences starting at position zero; no KV
cache is constructed. Captures are synchronous borrowed tensors [B,H,N,D],
with K/V heads still unduplicated. Copy them in the callback if retaining them.

Call release_current() between embedding, decoder and projection stages when
needed. The decoder cache contains exactly one layer, reused across variants;
close() releases it and the stream can subsequently load another layer.
"""

import inspect
from collections.abc import Callable
from pathlib import Path
from types import FunctionType, MethodType

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.llama import modeling_llama
from transformers.models.olmo2 import modeling_olmo2
from transformers.models.qwen2 import modeling_qwen2

from study.weights import SafeTensorWeights

Capture = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor, float], None]
_VARIANTS = {"bf16", "tile", "rotate", "smooth_k", "rotate_smooth_k", "smooth_kq"}
_ARCHITECTURES = {
    "qwen2": (modeling_qwen2, "Qwen2"),
    "llama": (modeling_llama, "Llama"),
    "olmo2": (modeling_olmo2, "Olmo2"),
}


class LayerStream:
    """Native decoder inference without constructing a complete model.

    ``embed(ids)`` -> repeated ``forward_layer(i, hidden)`` -> ``finish(hidden)``
    -> ``project_logits(hidden, start, stop)``. Logits are returned as float32,
    after the native projection in the stream dtype. Parent owns thread settings.
    """

    def __init__(self, snapshot: Path, dtype: torch.dtype = torch.bfloat16):
        self.config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        if self.config.model_type not in _ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {self.config.model_type}")
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("LayerStream supports CPU float32 or bfloat16")
        self.config._attn_implementation = "sdpa"
        self.dtype = dtype
        self.layer_count = self.config.num_hidden_layers
        self.weights = SafeTensorWeights(snapshot)
        self._architecture, name = _ARCHITECTURES[self.config.model_type]
        self._decoder_class = getattr(self._architecture, name + "DecoderLayer")
        self._norm_class = getattr(self._architecture, name + "RMSNorm")
        self._rotary = getattr(self._architecture, name + "RotaryEmbedding")(self.config)
        self._layer = None
        self._layer_index = None
        # Transformers may omit either member of a tied pair in safetensors.
        self._embedding_name = "model.embed_tokens.weight"
        self._head_name = "lm_head.weight"
        if self.config.tie_word_embeddings:
            if self._embedding_name not in self.weights:
                self._embedding_name = self._head_name
            if self._head_name not in self.weights:
                self._head_name = self._embedding_name

    def release_current(self) -> None:
        """Release the current owned decoder tensors; no file handles are cached."""
        self._layer = None
        self._layer_index = None

    def close(self) -> None:
        self.release_current()

    def _load_layer(self, index: int) -> torch.nn.Module:
        if not 0 <= index < self.layer_count:
            raise IndexError(f"Layer {index} outside [0, {self.layer_count})")
        if self._layer_index != index:
            # Release before allocation, never retaining two decoder layers.
            self.release_current()
            with torch.device("meta"):
                layer = self._decoder_class(self.config, index)
            self.weights.load_module(layer, f"model.layers.{index}.", self.dtype)
            self._layer = layer
            self._layer_index = index
        return self._layer

    @torch.inference_mode()
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.device.type != "cpu" or ids.dtype != torch.long or ids.ndim != 2:
            raise ValueError("Token IDs must be a CPU int64 tensor [B,N]")
        return self.weights.rows(self._embedding_name, ids, self.dtype)

    @torch.inference_mode()
    def forward_layer(
        self,
        index: int,
        hidden: torch.Tensor,
        variant: str = "bf16",
        capture: Capture | None = None,
    ) -> torch.Tensor:
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown attention variant: {variant}")
        if hidden.device.type != "cpu" or hidden.dtype != self.dtype or hidden.ndim != 3:
            raise ValueError("Hidden states must be CPU [B,N,H] in the stream dtype")
        layer = self._load_layer(index)
        sliding = getattr(layer, "attention_type", "full_attention") == "sliding_attention"
        if sliding and variant != "bf16":
            raise ValueError("FP8 variants support full causal attention, not sliding windows")
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        position_ids = positions.unsqueeze(0)
        position_embeddings = self._rotary(hidden, position_ids)
        mask_factory = create_sliding_window_causal_mask if sliding else create_causal_mask
        mask = mask_factory(
            config=self.config,
            input_embeds=hidden,
            attention_mask=None,
            cache_position=positions,
            past_key_values=None,
            position_ids=position_ids,
        )
        attention = layer.self_attn
        native_attention = self._architecture.ALL_ATTENTION_FUNCTIONS["sdpa"]

        def intercepted(module, query, key, value, attention_mask, scaling, **kwargs):
            if capture is not None:
                capture(index, query, key, value, scaling)
            if variant == "bf16":
                return native_attention(
                    module, query, key, value, attention_mask, scaling=scaling, **kwargs
                )
            from study.attention import apply_attention

            output = apply_attention(query, key, value, variant, scale=scaling)
            # Native attention expects [B,N,H,D], not the kernel's [B,H,N,D].
            return output.transpose(1, 2).contiguous().to(query.dtype), None

        intercept = capture is not None or variant != "bf16"
        if intercept:
            # Rebind only this instance's native forward global lookup. Its code,
            # projections, Q/K norms, RoPE and output reshape remain unchanged;
            # no process-global monkeypatch can affect another native model.
            forward = inspect.unwrap(type(attention).forward)
            globals_copy = dict(forward.__globals__, ALL_ATTENTION_FUNCTIONS={"sdpa": intercepted})
            rebound = FunctionType(
                forward.__code__,
                globals_copy,
                forward.__name__,
                forward.__defaults__,
                forward.__closure__,
            )
            rebound.__kwdefaults__ = forward.__kwdefaults__
            attention.forward = MethodType(rebound, attention)
        try:
            return layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=positions,
                position_embeddings=position_embeddings,
            )
        finally:
            if intercept:
                # Remove the instance override, including its bound-method cycle.
                del attention.forward

    @torch.inference_mode()
    def finish(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.device("meta"):
            norm = self._norm_class(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.weights.load_module(norm, "model.norm.", self.dtype)
        return norm(hidden)

    @torch.inference_mode()
    def project_logits(self, hidden: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        if not 0 <= start < stop <= self.config.vocab_size:
            raise ValueError("Vocabulary slice must satisfy 0 <= start < stop <= vocab_size")
        weight = self.weights.interval(self._head_name, start, stop, self.dtype)
        return F.linear(hidden, weight).float()
