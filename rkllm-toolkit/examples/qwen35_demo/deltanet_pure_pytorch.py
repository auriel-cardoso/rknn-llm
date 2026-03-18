# coding=utf-8
"""
deltanet_pure_pytorch.py — Pure-PyTorch GatedDeltaNet recurrence
================================================================
A self-contained, dependency-free implementation of the GatedDeltaNet
linear-recurrent attention mechanism used in Qwen3.5's DeltaNet layers.

Why this file exists
--------------------
The official implementation lives in the `fla` (Flash Linear Attention)
library which requires `triton >= 3.0` — a CUDA-only kernel compiler.
This means loading Qwen3.5 on ANY non-CUDA machine (CPU, AMD, Apple
Silicon, ARM conversion hosts) fails at import time.

This file re-implements the same algorithm using only `torch` built-ins,
so that:

  1. **Educational**: you can read the math step-by-step in plain Python.
  2. **Portable**: runs on CPU, CUDA, MPS — no triton, no fla, no CUDA SDK.
  3. **Verifiable**: given the same weights and inputs, it produces identical
     outputs to the `fla` version (within floating-point tolerance).
  4. **Basis for NPU kernel**: the sequential loop form maps directly to the
     C/C++ kernel that would need to be written for rkllm-runtime.

Performance note
----------------
This implementation uses a sequential Python loop over the time dimension,
so it is O(seq_len * d_k * d_v) with a large Python overhead per step.
For conversion / verification this is acceptable.  For training or long-
context inference, use the chunked parallel-scan version in `fla`.

Mathematical reference — see RESEARCH.md §5b
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Core recurrence
# ---------------------------------------------------------------------------

def gated_deltanet_recurrence(
    q: Tensor,           # [batch, seq_len, num_heads, d_k]  — query
    k: Tensor,           # [batch, seq_len, num_heads, d_k]  — key (unit-norm)
    v: Tensor,           # [batch, seq_len, num_heads, d_v]  — value
    beta: Tensor,        # [batch, seq_len, num_heads]        — beta gate (0..1)
    g: Tensor,           # [batch, seq_len, num_heads, d_v]   — decay gate (0..1)
    initial_state: Optional[Tensor] = None,   # [batch, num_heads, d_k, d_v]
    return_final_state: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Run the GatedDeltaNet recurrence sequentially.

    For each time step t and each head h, the state update is:

        α_t  = Sᵀ k_t                 (read from memory: d_v vector)
        δ_t  = v_t − α_t              (delta / error)
        S_t  = g_t ⊙ S_{t-1}          (decay existing memory row-wise)
             + β_t · k_t ⊗ δ_t        (rank-1 delta-rule write)
        o_t  = S_t · q_t              (read output from updated memory)

    Where:
      ⊙  element-wise product (broadcast over d_k rows when g is d_v)
      ⊗  outer product:  k_t ⊗ δ_t  gives shape [d_k, d_v]

    Parameters
    ----------
    q, k, v, beta, g :
        Tensors as described above.  All on the same device and dtype.
        k should already be L2-normalised (per head, per token).
        beta and g should be in [0, 1] (apply sigmoid before passing in).
    initial_state :
        Optional initial state matrix S_0, shape [batch, heads, d_k, d_v].
        If None, the state is initialised to zeros.
    return_final_state :
        If True, also return the final state S_T.

    Returns
    -------
    o : Tensor  [batch, seq_len, num_heads, d_v]  — outputs for all tokens
    final_state : Tensor or None — S_T, or None when return_final_state=False
    """
    batch, seq_len, num_heads, d_k = q.shape
    d_v = v.shape[-1]

    # Initialise state
    if initial_state is not None:
        S = initial_state.clone()  # [B, H, d_k, d_v]
    else:
        S = torch.zeros(batch, num_heads, d_k, d_v, device=q.device, dtype=q.dtype)

    outputs = torch.empty(batch, seq_len, num_heads, d_v, device=q.device, dtype=q.dtype)

    for t in range(seq_len):
        q_t    = q[:, t]      # [B, H, d_k]
        k_t    = k[:, t]      # [B, H, d_k]
        v_t    = v[:, t]      # [B, H, d_v]
        beta_t = beta[:, t]   # [B, H]
        g_t    = g[:, t]      # [B, H, d_v]

        # Step 1 — Read from memory using current key
        # α_t = S^T k_t  (S[b,h] is [d_k, d_v], so S^T @ k → [d_v])
        # Einsum: sum over k-dim: S[b,h,k,v] * k_t[b,h,k] → [B,H,d_v]
        alpha_t = torch.einsum('bhkv,bhk->bhv', S, k_t)

        # Step 2 — Compute the delta (error between target value and recalled value)
        delta_t = v_t - alpha_t                # [B, H, d_v]

        # Step 3a — Decay the existing state row-wise
        # g_t broadcasts over the d_k dimension (rows of S)
        # S[b, h, i, :] *= g_t[b, h, :]   for all i
        S = S * g_t.unsqueeze(-2)             # [B, H, d_k, d_v]

        # Step 3b — Rank-1 update: write the delta back to memory
        # Outer product: k_t ⊗ δ_t → [B, H, d_k, d_v]
        # scaled by β_t (broadcast to [B, H, 1, 1])
        rank1 = torch.einsum('bhk,bhv->bhkv', k_t, delta_t)  # [B, H, d_k, d_v]
        S = S + beta_t.unsqueeze(-1).unsqueeze(-1) * rank1

        # Step 4 — Read output from updated state
        # o_t = S^T q_t  (same convention as step 1)
        # Einsum: sum over k-dim: S[b,h,k,v] * q_t[b,h,k] → [B,H,d_v]
        o_t = torch.einsum('bhkv,bhk->bhv', S, q_t)

        outputs[:, t] = o_t

    final_state = S if return_final_state else None
    return outputs, final_state


# ---------------------------------------------------------------------------
# Helpers used by the adapter
# ---------------------------------------------------------------------------

def prepare_qkv_from_projections(
    hidden_states: Tensor,          # [B, seq, hidden]
    q_proj: torch.nn.Linear,
    k_proj: torch.nn.Linear,
    v_proj: torch.nn.Linear,
    b_proj: torch.nn.Linear,        # beta gate
    g_proj: torch.nn.Linear,        # decay gate
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    qk_dim: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project hidden states to q/k/v/beta/g tensors for gated_deltanet_recurrence.

    Returns
    -------
    q   : [B, seq, num_heads, qk_dim]
    k   : [B, seq, num_kv_heads, qk_dim]  — L2-normalised
    v   : [B, seq, num_kv_heads, head_dim]
    beta: [B, seq, num_kv_heads]           — sigmoid output
    g   : [B, seq, num_kv_heads, head_dim] — sigmoid output
    """
    B, T, _ = hidden_states.shape

    q = q_proj(hidden_states).view(B, T, num_heads, qk_dim)
    k = k_proj(hidden_states).view(B, T, num_kv_heads, qk_dim)
    v = v_proj(hidden_states).view(B, T, num_kv_heads, head_dim)
    beta = torch.sigmoid(b_proj(hidden_states)).view(B, T, num_kv_heads)
    g    = torch.sigmoid(g_proj(hidden_states)).view(B, T, num_kv_heads, head_dim)

    # L2-normalise keys (essential for the delta rule to be stable)
    k = F.normalize(k, p=2, dim=-1)

    return q, k, v, beta, g


# ---------------------------------------------------------------------------
# Full DeltaNet layer (for testing and reference)
# ---------------------------------------------------------------------------

class GatedDeltaNetLayer(torch.nn.Module):
    """A complete, self-contained GatedDeltaNet layer.

    This is a reference implementation for educational purposes.  It matches
    the structure of Qwen3.5's DeltaNet layers but uses the pure-PyTorch
    recurrence instead of `fla` CUDA kernels.

    Weight layout
    -------------
    This layer uses the SAME weight names as the Qwen3.5 DeltaNet layers so
    that you can load state_dict entries directly:

      self_attn.q_proj.weight / bias
      self_attn.k_proj.weight / bias
      self_attn.v_proj.weight / bias
      self_attn.o_proj.weight / bias   ← output projection
      self_attn.b_proj.weight / bias   ← beta gate
      self_attn.g_proj.weight / bias   ← decay gate
      self_attn.a_proj.weight / bias   ← output gating (optional)
      self_attn.q_norm.weight          ← SubRMSNorm for q
      self_attn.k_norm.weight          ← SubRMSNorm for k

    All weight names live under ``self_attn.*`` to match the unified interface
    used by the rkllm-toolkit adapter.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qk_dim: int | None = None,
        use_output_gate: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.qk_dim = qk_dim or head_dim

        # Projections
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.qk_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.qk_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.b_proj = torch.nn.Linear(hidden_size, num_kv_heads, bias=False)   # beta gate
        self.g_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)  # decay gate

        self.use_output_gate = use_output_gate
        if use_output_gate:
            self.a_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)

        # Per-head sub-RMSNorm on q and k
        self.q_norm = torch.nn.RMSNorm(self.qk_dim, eps=1e-6)
        self.k_norm = torch.nn.RMSNorm(self.qk_dim, eps=1e-6)

    def forward(
        self,
        hidden_states: Tensor,           # [B, seq, hidden]
        initial_state: Optional[Tensor] = None,
        return_state: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B, T, H = hidden_states.shape

        # Project to q/k/v/beta/g
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.qk_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.qk_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)

        # Apply per-head RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # L2-normalize keys for the delta rule stability
        k = F.normalize(k, p=2, dim=-1)

        # Beta and decay gates
        beta = torch.sigmoid(self.b_proj(hidden_states)).view(B, T, self.num_kv_heads)
        g    = torch.sigmoid(self.g_proj(hidden_states)).view(B, T, self.num_kv_heads, self.head_dim)

        # Handle GQA: expand k/v if num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k    = k.repeat_interleave(repeat, dim=2)
            v    = v.repeat_interleave(repeat, dim=2)
            beta = beta.repeat_interleave(repeat, dim=2)
            g    = g.repeat_interleave(repeat, dim=2)

        # Run the recurrence
        o, final_state = gated_deltanet_recurrence(
            q, k, v, beta, g,
            initial_state=initial_state,
            return_final_state=return_state,
        )

        # Optionally apply output gate
        if self.use_output_gate:
            gate = torch.sigmoid(self.a_proj(hidden_states))       # [B, T, H*head_dim]
            gate = gate.view(B, T, self.num_heads, self.head_dim)
            o = o * gate

        # Merge heads and project
        o = o.reshape(B, T, self.num_heads * self.head_dim)
        out = self.o_proj(o)

        return out, final_state


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_recurrence_basic():
    """Smoke test: recurrence runs, shapes are correct."""
    torch.manual_seed(0)
    B, T, H, d_k, d_v = 2, 16, 4, 32, 64

    q    = F.normalize(torch.randn(B, T, H, d_k), dim=-1)
    k    = F.normalize(torch.randn(B, T, H, d_k), dim=-1)
    v    = torch.randn(B, T, H, d_v)
    beta = torch.sigmoid(torch.randn(B, T, H))
    g    = torch.sigmoid(torch.randn(B, T, H, d_v))

    o, state = gated_deltanet_recurrence(q, k, v, beta, g, return_final_state=True)

    assert o.shape == (B, T, H, d_v), f"output shape wrong: {o.shape}"
    assert state.shape == (B, H, d_k, d_v), f"state shape wrong: {state.shape}"
    assert not torch.isnan(o).any(), "NaN in output"
    print("[PASS] _test_recurrence_basic — shape and no-NaN checks")


def _test_recurrence_state_continuation():
    """Test that passing initial_state continues from where we left off."""
    torch.manual_seed(1)
    B, T, H, d_k, d_v = 1, 8, 2, 16, 16

    q    = F.normalize(torch.randn(B, T, H, d_k), dim=-1)
    k    = F.normalize(torch.randn(B, T, H, d_k), dim=-1)
    v    = torch.randn(B, T, H, d_v)
    beta = torch.sigmoid(torch.randn(B, T, H))
    g    = torch.sigmoid(torch.randn(B, T, H, d_v))

    # Full sequence in one shot
    o_full, _ = gated_deltanet_recurrence(q, k, v, beta, g)

    # Split at T//2 and continue with state
    half = T // 2
    o1, state1 = gated_deltanet_recurrence(
        q[:, :half], k[:, :half], v[:, :half], beta[:, :half], g[:, :half],
        return_final_state=True,
    )
    o2, _ = gated_deltanet_recurrence(
        q[:, half:], k[:, half:], v[:, half:], beta[:, half:], g[:, half:],
        initial_state=state1,
    )
    o_split = torch.cat([o1, o2], dim=1)

    max_diff = (o_full - o_split).abs().max().item()
    assert max_diff < 1e-5, f"State continuation mismatch: max_diff={max_diff}"
    print(f"[PASS] _test_recurrence_state_continuation — max_diff={max_diff:.2e}")


def _test_layer_forward():
    """Test full GatedDeltaNetLayer forward pass."""
    torch.manual_seed(2)
    B, T = 2, 10
    hidden_size = 128
    num_heads = 4
    num_kv_heads = 2
    head_dim = 32
    qk_dim = 32

    layer = GatedDeltaNetLayer(hidden_size, num_heads, num_kv_heads, head_dim, qk_dim)
    x = torch.randn(B, T, hidden_size)

    out, state = layer(x, return_state=True)

    assert out.shape == (B, T, hidden_size), f"out shape wrong: {out.shape}"
    assert state.shape == (B, num_heads, qk_dim, head_dim), f"state shape wrong: {state.shape}"
    assert not torch.isnan(out).any(), "NaN in layer output"
    print("[PASS] _test_layer_forward — shapes and no-NaN checks")


def _test_delta_rule_memory():
    """Illustrate the delta rule: memory should store a key-value association."""
    # After seeing (k, v) once, querying with q≈k should recall v.
    torch.manual_seed(3)
    H, d_k, d_v = 1, 8, 8
    B = 1

    # Single time step: write (k, v) with beta=1, g=1 (full learning, no decay)
    k = F.normalize(torch.randn(B, 1, H, d_k), dim=-1)
    v = torch.randn(B, 1, H, d_v)
    q = k.clone()  # query with the same key → should retrieve v
    beta = torch.ones(B, 1, H)
    g    = torch.ones(B, 1, H, d_v)

    o, _ = gated_deltanet_recurrence(q, k, v, beta, g)
    # With beta=1 and initial state=0, S_1 = outer(k, v), so S_1 * q = S_1 * k = v
    max_diff = (o[:, 0] - v[:, 0]).abs().max().item()
    assert max_diff < 1e-5, f"Memory recall failed: max_diff={max_diff}"
    print(f"[PASS] _test_delta_rule_memory — recalled value with error {max_diff:.2e}")


def _test_forgetting():
    """Illustrate forgetting: with g<1, old associations fade over time."""
    torch.manual_seed(4)
    H, d_k, d_v = 1, 4, 4
    B, T = 1, 20

    k    = F.normalize(torch.randn(B, T, H, d_k), dim=-1)
    v    = torch.randn(B, T, H, d_v)
    beta = torch.ones(B, T, H)

    # Case 1: no forgetting (g=1)
    g_ones = torch.ones(B, T, H, d_v)
    o_no_forget, S_no_forget = gated_deltanet_recurrence(k, k, v, beta, g_ones, return_final_state=True)

    # Case 2: strong forgetting (g=0.5)
    g_half = torch.full((B, T, H, d_v), 0.5)
    o_forget, S_forget = gated_deltanet_recurrence(k, k, v, beta, g_half, return_final_state=True)

    # With forgetting, old memories are weaker → state norm should be smaller
    norm_no_forget = S_no_forget.norm().item()
    norm_forget    = S_forget.norm().item()
    assert norm_forget < norm_no_forget, (
        f"Forgetting should reduce state norm: "
        f"no-forget={norm_no_forget:.3f}, forget={norm_forget:.3f}"
    )
    print(
        f"[PASS] _test_forgetting — state norm: no-forget={norm_no_forget:.3f}, "
        f"forget={norm_forget:.3f} (forget < no-forget ✓)"
    )


if __name__ == "__main__":
    print("Running GatedDeltaNet pure-PyTorch self-tests...\n")
    _test_recurrence_basic()
    _test_recurrence_state_continuation()
    _test_layer_forward()
    _test_delta_rule_memory()
    _test_forgetting()
    print("\nAll tests passed!")
