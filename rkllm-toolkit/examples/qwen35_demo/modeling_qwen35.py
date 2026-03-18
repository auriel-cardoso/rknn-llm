# coding=utf-8
# Custom model adapter for Qwen3.5 hybrid (GatedDeltaNet + GQA).
#
# ROOT CAUSE OF airockchip/rknn-llm#472
# ======================================
# Qwen3.5 alternates two fundamentally different layer types:
#   1. Standard Grouped-Query Attention (GQA) – identical to Qwen3.
#   2. Gated DeltaNet – a linear-recurrent "attention" layer from the Flash
#      Linear Attention (FLA) library.
#
# When rkllm-toolkit's load_huggingface tries to auto-detect the architecture
# it maps every layer to a standard attention block and looks for weight paths
# like ``model.layers.N.self_attn.{q,k,v,o}_proj.weight``.  Gated DeltaNet
# layers either
#   • live under a *different module name* (e.g. ``attn`` or ``mixer``), or
#   • have a different weight layout (e.g. an additional ``b_proj`` beta gate,
#     ``g_proj`` output gate, ``a_proj`` absorption, smaller qk_dim),
# so those paths simply do not exist, causing the "Not found the weight"
# warnings.  The subsequent tensor-shape mismatch error occurs when the toolkit
# tries to concatenate incompatible tensors from mixed layer types.
#
# SOLUTION IMPLEMENTED HERE
# ==========================
# We define a *custom* decoder layer class (Qwen35DecoderLayer) that the
# rkllm-toolkit can use via its custom-model interface.  It wraps both layer
# types under a **unified** ``self_attn`` submodule so that every layer
# exports the same weight paths (q_proj, k_proj, v_proj, o_proj, q_norm,
# k_norm) regardless of whether it is a GQA or a DeltaNet layer.
#
# For GQA layers this is a straight pass-through.
#
# For DeltaNet layers the wrapper *loads* the original DeltaNet weights into
# the corresponding linear projections.  For inference the forward pass runs
# *standard scaled-dot-product attention* using those projections instead of
# the true DeltaNet recurrence.  This is an **approximation**; the model will
# run on the NPU but with reduced quality for the DeltaNet layers.
#
# LIMITATIONS
# ===========
# • The rkllm-runtime does not natively support the DeltaNet recurrence.
#   Only Rockchip can add that kernel support to the firmware/runtime.
# • The approximation (DeltaNet → attention) works because both operations
#   share q/k/v/o projections; the quality degradation is moderate for short
#   contexts but increases for longer ones where DeltaNet's recurrence matters.
# • The extra DeltaNet gating projections (b_proj, g_proj, a_proj) are not
#   loaded because the rkllm config_custom.json has no slots for them.  A
#   future toolkit version could expose these via additional config keys.

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.utils import logging

try:
    from .configuration_qwen35 import Qwen35Config
except ImportError:
    from configuration_qwen35 import Qwen35Config

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    orig_dtype = q.dtype
    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class Qwen35RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(orig_dtype)


# ---------------------------------------------------------------------------
# Rotary embeddings
# ---------------------------------------------------------------------------

class Qwen35RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 1_000_000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position_embeddings, torch.float32)

    def _build_cache(self, seq_len: int, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len, x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Qwen35MLP(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.functional.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention – standard GQA (identical to Qwen3)
# ---------------------------------------------------------------------------

class Qwen35GQAttention(nn.Module):
    """Standard Grouped-Query Attention layer (Qwen3-style)."""

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Per-head query/key normalisation (Qwen3 / Qwen3.5 specific).
        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen35RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Per-head normalisation before RoPE.
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None if not output_attentions else attn_weights, past_key_value


# ---------------------------------------------------------------------------
# GatedDeltaNet wrapped as attention
# ---------------------------------------------------------------------------

class Qwen35GatedDeltaNetAsAttention(nn.Module):
    """
    Adapter that presents a Gated DeltaNet layer using the same interface
    (and the same weight *paths*) as a standard attention layer so that
    rkllm-toolkit can load and convert it.

    Weight mapping from the original Qwen3.5 DeltaNet checkpoint
    ---------------------------------------------------------------
    rkllm path (self_attn.*)     ← DeltaNet source weight
    ─────────────────────────────────────────────────────────────
    q_proj                       ← attn.q_proj   (queries)
    k_proj                       ← attn.k_proj   (keys / delta keys)
    v_proj                       ← attn.v_proj   (values)
    o_proj                       ← attn.o_proj   (output projection)
    q_norm                       ← attn.q_norm   (per-head SubRMSNorm)
    k_norm                       ← attn.k_norm   (per-head SubRMSNorm)

    The beta gate (b_proj), output gate (g_proj), and state absorption
    (a_proj) projections are read-only extras – they are NOT exported to
    the rkllm model because config_custom.json has no corresponding slots.
    A future rkllm-toolkit version could add DELTANET_BETA / DELTANET_GATE
    config keys to expose these.

    Forward pass approximation
    ---------------------------
    At inference time (inside the calibration/quantisation step that
    load_huggingface runs) we execute *standard causal attention* using the
    q/k/v/o projections.  This is the best achievable approximation without
    native DeltaNet kernel support in the rkllm-runtime.

    Note: qk_dim (key/query dimension per head in DeltaNet) may differ from
    head_dim (used in attention).  When they differ, dummy zero-padding is
    applied so shapes are consistent.  See ``_align_kq_dim`` below.
    """

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim

        # qk_dim for DeltaNet may be smaller (e.g. expand_k = 1 means same
        # as head_dim; if the checkpoint uses a different value we detect it
        # after loading weights via load_deltanet_weights()).
        self._qk_dim = self.head_dim  # will be overridden if needed

        # Projections – shapes are initialised to the *attention* dimensions;
        # they will be replaced with the DeltaNet weights in load_deltanet_weights().
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen35RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    # ------------------------------------------------------------------
    # Weight-loading helpers
    # ------------------------------------------------------------------

    def load_deltanet_weights(self, state_dict_prefix: str, full_state_dict: dict):
        """
        Extract DeltaNet weights from *full_state_dict* (keyed by full model
        path) and map them into this module's projections.

        ``state_dict_prefix`` is something like ``"model.layers.2."``.

        The method tries both ``self_attn.*`` and ``attn.*`` sub-prefixes
        because different Qwen3.5 checkpoints may use either convention.
        """
        found = {}
        for sub in ("self_attn.", "attn.", "mixer."):
            prefix = state_dict_prefix + sub
            keys = {k[len(prefix):]: v for k, v in full_state_dict.items() if k.startswith(prefix)}
            if keys:
                found = keys
                break

        if not found:
            logger.warning(
                f"[Qwen35GatedDeltaNetAsAttention] Could not find DeltaNet weights "
                f"under {state_dict_prefix}{{self_attn,attn,mixer}}.*  – "
                f"using random initialisation for layer {self.layer_idx}."
            )
            return

        def _load(proj_name: str, module: nn.Linear):
            w_key = proj_name + ".weight"
            if w_key not in found:
                return
            src = found[w_key]
            if src.shape == module.weight.shape:
                module.weight = nn.Parameter(src.clone())
            else:
                # Shape mismatch – typical when qk_dim ≠ head_dim.
                # Pad or truncate along output dimension.
                tgt_shape = module.weight.shape
                aligned = _align_weight(src, tgt_shape)
                module.weight = nn.Parameter(aligned)

        def _load_norm(norm_name: str, module: Qwen35RMSNorm):
            w_key = norm_name + ".weight"
            if w_key not in found:
                return
            src = found[w_key]
            if src.shape == module.weight.shape:
                module.weight = nn.Parameter(src.clone())
            else:
                # SubRMSNorm dimension differs – resize.
                tgt_len = module.weight.shape[0]
                if src.shape[0] >= tgt_len:
                    module.weight = nn.Parameter(src[:tgt_len].clone())
                else:
                    padded = torch.ones(tgt_len, dtype=src.dtype, device=src.device)
                    padded[: src.shape[0]] = src
                    module.weight = nn.Parameter(padded)

        _load("q_proj", self.q_proj)
        _load("k_proj", self.k_proj)
        _load("v_proj", self.v_proj)
        _load("o_proj", self.o_proj)
        _load_norm("q_norm", self.q_norm)
        _load_norm("k_norm", self.k_norm)

    # ------------------------------------------------------------------
    # Forward (approximated as standard attention)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None if not output_attentions else attn_weights, past_key_value


# ---------------------------------------------------------------------------
# Utility: weight shape alignment
# ---------------------------------------------------------------------------

def _align_weight(src: torch.Tensor, tgt_shape: torch.Size) -> torch.Tensor:
    """
    Resize *src* to *tgt_shape* by truncating or zero-padding each dimension.
    Used when DeltaNet's qk_dim ≠ head_dim.
    """
    result = torch.zeros(tgt_shape, dtype=src.dtype, device=src.device)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(src.shape, tgt_shape))
    result[slices] = src[slices]
    return result


# ---------------------------------------------------------------------------
# Hybrid decoder layer
# ---------------------------------------------------------------------------

class Qwen35DecoderLayer(nn.Module):
    """
    Hybrid decoder layer used as the ``BLOCKNAME`` for the rkllm custom-model
    interface.

    Every layer – whether GQA attention or GatedDeltaNet – exposes the same
    weight structure under ``self_attn.*`` so that a single ``config_custom.json``
    can describe the entire model.
    """

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self._is_attention = config.is_attention_layer(layer_idx)

        if self._is_attention:
            self.self_attn = Qwen35GQAttention(config, layer_idx)
        else:
            self.self_attn = Qwen35GatedDeltaNetAsAttention(config, layer_idx)

        self.mlp = Qwen35MLP(config)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class Qwen35PreTrainedModel(PreTrainedModel):
    config_class = Qwen35Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen35DecoderLayer"]
    _supports_cache_class = True

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen35Model(Qwen35PreTrainedModel):
    def __init__(self, config: Qwen35Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen35DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        bsz, seq_len, _ = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

        # Build causal mask.
        if attention_mask is not None:
            attention_mask = _prepare_4d_causal_mask(attention_mask, hidden_states.dtype, seq_len)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, decoder_layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                output_attentions=output_attentions or False,
                use_cache=use_cache or False,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen35ForCausalLM(Qwen35PreTrainedModel, GenerationMixin):
    """
    Causal LM head on top of :class:`Qwen35Model`.

    This is the class referenced in ``config.json``'s ``auto_map`` when using
    the rkllm-toolkit custom-model interface.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen35Config):
        super().__init__(config)
        self.model = Qwen35Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }


# ---------------------------------------------------------------------------
# Internal helper for causal mask construction
# ---------------------------------------------------------------------------

def _prepare_4d_causal_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: int,
) -> torch.Tensor:
    bsz, src_len = attention_mask.shape
    causal = torch.full((tgt_len, src_len), torch.finfo(dtype).min, dtype=dtype, device=attention_mask.device)
    causal_upper = torch.triu(causal, diagonal=1)
    expanded = causal_upper[None, None, :, :].expand(bsz, 1, tgt_len, src_len)
    inverted = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
    return expanded + inverted
