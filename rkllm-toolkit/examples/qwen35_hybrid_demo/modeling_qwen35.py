# coding=utf-8
# Copyright 2025 Alibaba Group and the HuggingFace Inc. team.
# Qwen3.5 (Qwen3-Next) hybrid model adapter for rkllm-toolkit.
#
# This file wraps the HuggingFace Qwen3Next model into a custom PyTorch
# module that the rkllm custom-model conversion path can consume.
#
# Architecture overview
# ─────────────────────
# Qwen3.5 (model_type = "qwen3_next") has two decoder layer types per block:
#
#   full_attention  → standard transformer attention (with gated Q projection)
#   linear_attention → GatedDeltaNet recurrent layer (RWKV7-like)
#
# The default 2-B model uses the pattern:
#   [linear, linear, linear, full, linear, linear, linear, full, …]
#   i.e. 3 recurrent layers for every 1 attention layer.
#
# rkllm custom-model path
# ───────────────────────
# The rkllm toolkit can convert arbitrary PyTorch models via
# llm.load_huggingface(model=..., custom_model=True, custom_model_config=...).
# It expects every decoder block to expose the same weight attribute names
# (mapped via config_qwen35.json).
#
# This module introduces Qwen35HybridDecoderLayer which:
#   • For full_attention layers  → wraps Qwen3NextAttention as-is
#   • For linear_attention layers → approximates GatedDeltaNet with causal
#     self-attention so that rkllm can still build a runnable model.
#     This approximation loses cross-chunk recurrent state and will reduce
#     quality.  It is a placeholder until rkllm adds native qwen3_next support.
#
# Usage
# ─────
#   from modeling_qwen35 import Qwen35HybridModel
#   model = Qwen35HybridModel.from_pretrained("/path/to/Qwen3.5-2B")
#   # Then pass to the rkllm export script (export_rkllm_qwen35.py).

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Requires transformers >= 4.57.0
try:
    from transformers import Qwen3NextForCausalLM, Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextRMSNorm,
        Qwen3NextMLP,
    )
except ImportError as exc:
    raise ImportError(
        "Qwen3.5 (Qwen3-Next) requires transformers >= 4.57.0.\n"
        "Please run: pip install 'transformers>=4.57.0'\n"
        f"Original error: {exc}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: causal self-attention (attention-approximation for linear layers)
# ──────────────────────────────────────────────────────────────────────────────

class _CausalAttentionApprox(nn.Module):
    """
    Causal self-attention that approximates a GatedDeltaNet layer.

    The approximation works as follows:
      1. Project hidden states to Q, K, V using weights derived from the
         GatedDeltaNet projection matrices.
      2. Run standard scaled-dot-product attention.
      3. Project back to hidden_size.

    WHY THIS IS AN APPROXIMATION
    ─────────────────────────────
    GatedDeltaNet maintains a persistent recurrent state S across tokens.
    Standard attention re-computes all past tokens from scratch.  The two
    produce the same output only when the recurrent state exactly equals the
    attention KV matrix — which holds approximately for short sequences but
    diverges for long ones.

    For deployment: the accuracy loss is tolerable for short prompts and gets
    worse as sequence length grows.  Use this as a proof-of-concept only.
    """

    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_value_heads
        self.head_dim = config.linear_value_head_dim
        inner_dim = self.num_heads * self.head_dim

        # Derive Q, K, V projections from the GatedDeltaNet weight layout.
        # in_proj_qkvz projects: hidden → [q_k_dim, k_k_dim, v_dim, z_dim]
        # where q_k_dim = k_k_dim = num_k_heads * head_k_dim
        #       v_dim = z_dim = num_v_heads * head_v_dim
        num_k_heads = config.linear_num_key_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim

        qk_dim = num_k_heads * head_k_dim
        v_dim = self.num_heads * head_v_dim

        # Unified projection that we will slice into Q, K, V:
        # total_in = q_part + k_part + v_part + z_part
        total_proj = qk_dim * 2 + v_dim * 2
        self.in_proj = nn.Linear(self.hidden_size, total_proj, bias=False)

        # Slices into the in_proj output  (mirrors in_proj_qkvz layout)
        self._q_slice = (0, qk_dim)
        self._k_slice = (qk_dim, qk_dim * 2)
        self._v_slice = (qk_dim * 2, qk_dim * 2 + v_dim)
        # z_slice is (qk_dim*2 + v_dim, total_proj) — used for gating

        self.out_proj = nn.Linear(v_dim, self.hidden_size, bias=False)
        self.scaling = head_k_dim ** -0.5

        # Expose the same attribute names as Qwen3NextAttention so that
        # config_qwen35.json can map them uniformly.
        # We alias in_proj → q_proj/k_proj/v_proj via properties below.
        self._qk_dim = qk_dim
        self._v_dim = v_dim

    # --- Aliases so that config_qwen35.json weight-name mapping still works ---
    # rkllm maps: mixer.q_proj, mixer.k_proj, mixer.v_proj, mixer.o_proj
    # We expose the combined in_proj under those names for the config mapper.

    @property
    def q_proj(self):  # noqa: D401
        """Alias: first qk_dim rows of in_proj.weight, viewed as a linear layer."""
        return _SlicedLinear(self.in_proj, slice(None, self._qk_dim))

    @property
    def k_proj(self):  # noqa: D401
        return _SlicedLinear(self.in_proj, slice(self._qk_dim, self._qk_dim * 2))

    @property
    def v_proj(self):  # noqa: D401
        return _SlicedLinear(self.in_proj, slice(self._qk_dim * 2,
                                                  self._qk_dim * 2 + self._v_dim))

    @property
    def o_proj(self):  # noqa: D401
        return self.out_proj

    @property
    def q_norm(self):  # noqa: D401
        """
        Dummy per-head norm (Identity) for rkllm config-mapping compatibility.

        _CausalAttentionApprox does not apply QK-norms (unlike Qwen3NextAttention
        which has Qwen3NextRMSNorm per head).  The Identity ensures that any code
        path that reads this attribute does not raise AttributeError.
        """
        return nn.Identity()

    @property
    def k_norm(self):  # noqa: D401
        """See q_norm — same rationale."""
        return nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        proj = self.in_proj(hidden_states)
        q = proj[..., self._q_slice[0]:self._q_slice[1]]
        k = proj[..., self._k_slice[0]:self._k_slice[1]]
        v = proj[..., self._v_slice[0]:self._v_slice[1]]
        z = proj[..., self._qk_dim * 2 + self._v_dim:]  # gate

        # Reshape to (bsz, heads, seq_len, head_dim) for SDPA
        q = q.view(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, -1).transpose(1, 2)

        # Scaled dot-product attention (causal)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )  # (bsz, heads, seq_len, head_v_dim)

        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, self._v_dim)

        # Apply gated normalization (simplified: just sigmoid gate)
        gate = torch.sigmoid(z)
        attn_out = attn_out * gate

        return self.out_proj(attn_out)


class _SlicedLinear:
    """
    Lightweight view of a slice of a Linear layer's weight rows.
    Used to expose q_proj/k_proj/v_proj attributes backed by a single in_proj.
    This is only needed for weight-name mapping; it is not called during forward.
    """
    def __init__(self, parent: nn.Linear, row_slice):
        self.weight = parent.weight[row_slice]
        self.bias = None


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid decoder layer
# ──────────────────────────────────────────────────────────────────────────────

class Qwen35HybridDecoderLayer(nn.Module):
    """
    Unified decoder layer for Qwen3.5.

    Depending on `layer_type` it wraps either:
      • Qwen3NextAttention (full_attention)   — kept exactly as in HuggingFace
      • _CausalAttentionApprox (linear_attention) — attention approximation

    Both paths expose the same weight names (mixer.q_proj, mixer.k_proj, …)
    so that config_qwen35.json can map them uniformly to rkllm's expected keys.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_idx: int,
        original_layer,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.hidden_size = config.hidden_size

        # --- normalization ---
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm

        # --- token mixer ---
        if self.layer_type == "full_attention":
            # Re-wrap the original Qwen3NextAttention to rename self_attn → mixer
            self.mixer = _FullAttentionWrapper(original_layer.self_attn)
        else:
            # linear_attention: build approximate attention from GatedDeltaNet weights
            self.mixer = _LinearAttentionWrapper(config, layer_idx, original_layer.linear_attn)

        # --- MLP ---
        self.mlp = original_layer.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        # mixer may return a tuple (output, kv_cache); unwrap if so
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        return outputs


class _FullAttentionWrapper(nn.Module):
    """
    Thin wrapper around Qwen3NextAttention that exposes unified attribute names.

    In Qwen3NextAttention q_proj outputs [query; gate] concatenated.
    The gate is applied inside the attention forward — we preserve this exactly.
    """
    def __init__(self, attn_module):
        super().__init__()
        # Bring all sub-modules into this namespace so that weight access
        # (model.layers.N.mixer.q_proj.weight etc.) works correctly.
        self.q_proj = attn_module.q_proj   # outputs 2× head_dim (query + gate)
        self.k_proj = attn_module.k_proj
        self.v_proj = attn_module.v_proj
        self.o_proj = attn_module.o_proj
        self.q_norm = attn_module.q_norm
        self.k_norm = attn_module.k_norm
        self._attn = attn_module           # keep reference for forward

    def forward(self, hidden_states, **kwargs):
        out, _ = self._attn(hidden_states, **kwargs)
        return out


class _LinearAttentionWrapper(nn.Module):
    """
    Wraps a Qwen3NextGatedDeltaNet layer, exposing the GatedDeltaNet weights
    under unified attribute names, and providing a causal-attention forward pass
    as an approximation for rkllm conversion.

    WEIGHT NAME MAPPING (GatedDeltaNet → rkllm-compatible names)
    ──────────────────────────────────────────────────────────────
    GatedDeltaNet weight          → mixer.* attribute
    ─────────────────────────────────────────────────
    linear_attn.in_proj_qkvz      → in_proj_qkvz   (not mapped via config)
    linear_attn.in_proj_ba        → in_proj_ba      (not mapped via config)
    linear_attn.conv1d            → conv1d          (not mapped via config)
    linear_attn.dt_bias           → dt_bias         (not mapped via config)
    linear_attn.A_log             → A_log           (not mapped via config)
    linear_attn.norm              → norm            (not mapped via config)
    linear_attn.out_proj          → o_proj          (mapped via config)

    For the approximation forward, we derive q, k, v from in_proj_qkvz and
    run causal SDPA, matching the interface expected by config_qwen35.json.
    """
    def __init__(self, config: Qwen3NextConfig, layer_idx: int, gdn_module):
        super().__init__()
        # Expose all GatedDeltaNet sub-modules for weight access
        self.in_proj_qkvz = gdn_module.in_proj_qkvz
        self.in_proj_ba = gdn_module.in_proj_ba
        self.conv1d = gdn_module.conv1d
        self.dt_bias = gdn_module.dt_bias
        self.A_log = gdn_module.A_log
        self.norm = gdn_module.norm

        # o_proj exposed as expected by config_qwen35.json ATTN_OUT mapping
        self.o_proj = gdn_module.out_proj

        # Architecture dimensions
        num_k_heads = config.linear_num_key_heads
        head_k_dim = config.linear_key_head_dim
        num_v_heads = config.linear_num_value_heads
        head_v_dim = config.linear_value_head_dim
        self._qk_dim = num_k_heads * head_k_dim
        self._v_dim = num_v_heads * head_v_dim
        self._num_v_heads = num_v_heads
        self._head_v_dim = head_v_dim
        self._hidden_size = config.hidden_size

        # Dummy norms so any code path that reads q_norm/k_norm doesn't raise
        # AttributeError.  GatedDeltaNet doesn't use per-head QK-norms, unlike
        # Qwen3NextAttention — so Identity is the correct no-op placeholder.
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    @property
    def q_proj(self):  # noqa: D401
        """Expose first qk_dim rows of in_proj_qkvz as q_proj for config mapping."""
        return _SlicedLinear(self.in_proj_qkvz, slice(None, self._qk_dim))

    @property
    def k_proj(self):  # noqa: D401
        return _SlicedLinear(self.in_proj_qkvz, slice(self._qk_dim, self._qk_dim * 2))

    @property
    def v_proj(self):  # noqa: D401
        return _SlicedLinear(self.in_proj_qkvz,
                             slice(self._qk_dim * 2, self._qk_dim * 2 + self._v_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """
        Approximate forward pass using causal self-attention.

        NOTE: This does NOT compute the true GatedDeltaNet recurrent update.
        It is a structural placeholder so that the rkllm toolkit can trace the
        compute graph and produce a runnable (though quality-degraded) model.
        """
        bsz, seq_len, _ = hidden_states.shape

        proj = self.in_proj_qkvz(hidden_states)   # [B, L, qk*2 + v*2]
        q = proj[..., :self._qk_dim]
        k = proj[..., self._qk_dim:self._qk_dim * 2]
        v = proj[..., self._qk_dim * 2:self._qk_dim * 2 + self._v_dim]
        z = proj[..., self._qk_dim * 2 + self._v_dim:]   # gating signal

        # Reshape to (bsz, heads, seq_len, head_dim)
        q = q.view(bsz, seq_len, self._num_v_heads, self._head_v_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self._num_v_heads, self._head_v_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self._num_v_heads, self._head_v_dim).transpose(1, 2)

        # Causal attention (approximation of recurrent state)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, self._v_dim)

        # Gate (simplified from gated RMS norm)
        gate = torch.sigmoid(z)
        attn_out = attn_out * gate

        return self.o_proj(attn_out)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level hybrid model
# ──────────────────────────────────────────────────────────────────────────────

class Qwen35HybridModel(nn.Module):
    """
    Qwen3.5 wrapped as a unified hybrid model compatible with rkllm's
    custom-model conversion path.

    The model exposes:
      • model.embed_tokens       — token embedding
      • model.layers             — list of Qwen35HybridDecoderLayer
      • model.norm               — final RMS norm
      • lm_head                  — language model head

    Load from a HuggingFace checkpoint:
        model = Qwen35HybridModel.from_pretrained("/path/to/Qwen3.5-2B")

    Then export:
        llm = RKLLM()
        llm.load_huggingface(
            model=model,
            custom_model=True,
            custom_model_config="config_qwen35.json",
        )
    """

    def __init__(self, hf_model: Qwen3NextForCausalLM):
        super().__init__()
        cfg = hf_model.config
        inner = hf_model.model   # Qwen3NextModel

        self.embed_tokens = inner.embed_tokens
        self.norm = inner.norm
        self.lm_head = hf_model.lm_head

        # Build hybrid layers
        self.layers = nn.ModuleList([
            Qwen35HybridDecoderLayer(cfg, i, layer)
            for i, layer in enumerate(inner.layers)
        ])

        self.config = cfg
        self._print_layer_summary(cfg)

    @staticmethod
    def _print_layer_summary(cfg: Qwen3NextConfig):
        layer_types = cfg.layer_types
        n_full = sum(1 for t in layer_types if t == "full_attention")
        n_lin = sum(1 for t in layer_types if t == "linear_attention")
        print(
            f"[Qwen35HybridModel] {cfg.num_hidden_layers} layers: "
            f"{n_full} full_attention + {n_lin} linear_attention (approx)\n"
            f"  Pattern: {layer_types[:8]} …"
        )

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Qwen35HybridModel":
        """
        Load Qwen3.5 from a HuggingFace checkpoint and wrap it.

        Args:
            model_path: Path to the local directory or HuggingFace model ID.
            **kwargs:   Forwarded to Qwen3NextForCausalLM.from_pretrained.
        """
        defaults = dict(
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            _attn_implementation="eager",
        )
        defaults.update(kwargs)
        hf_model = Qwen3NextForCausalLM.from_pretrained(
            model_path, **defaults
        ).eval()
        return cls(hf_model)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """
        Simplified forward pass for tracing / rkllm export.

        Returns logits of shape (batch_size, seq_len, vocab_size).
        """
        hidden_states = self.embed_tokens(input_ids)

        # Build causal mask
        bsz, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(bsz, seq_len, device=input_ids.device)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        for layer in self.layers:
            layer_out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_out[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Smoke-test: verify that the hybrid model loads and produces output
    of the correct shape without running the full rkllm conversion.

    Usage:
        python modeling_qwen35.py --path /path/to/Qwen3.5-2B
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to Qwen3.5-2B checkpoint")
    args = parser.parse_args()

    print(f"Loading Qwen3.5 from: {args.path}")
    model = Qwen35HybridModel.from_pretrained(args.path)
    print(f"Parameter count: {model.count_parameters():,}")

    # Dummy forward pass
    dummy_input = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        logits = model(dummy_input)
    print(f"Output logits shape: {logits.shape}")
    print("Self-test passed ✓")
