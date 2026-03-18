# Research: Qwen3.5 on rkllm — The Transformers Library Problem and the RWKV7 Comparison

> **Audience:** Community contributors wanting to understand the root cause and contribute
> human-made code.  This document answers the two key questions from
> [airockchip/rknn-llm#471](https://github.com/airockchip/rknn-llm/issues/471):
>
> 1. *"The official developer says the problem is the Transformers Python library — what does that mean exactly?"*
> 2. *"The framework already supports RWKV7 (an RNN with linear attention) — so why can't it support Qwen3.5?"*

---

## Table of Contents

1. [Problem Statement Recap](#1-problem-statement-recap)
2. [Layer 1 — The Transformers Library Problem](#2-layer-1--the-transformers-library-problem)
3. [Layer 2 — The rkllm-runtime Problem](#3-layer-2--the-rkllm-runtime-problem)
4. [RWKV7 vs GatedDeltaNet — Why One Works and the Other Doesn't](#4-rwkv7-vs-gateddeltanet--why-one-works-and-the-other-doesnt)
5. [The Mathematics of Linear Recurrent Attention](#5-the-mathematics-of-linear-recurrent-attention)
6. [What the Community CAN Do Right Now](#6-what-the-community-can-do-right-now)
7. [What Only Rockchip Can Do](#7-what-only-rockchip-can-do)
8. [A Roadmap for Your Own Contribution](#8-a-roadmap-for-your-own-contribution)

---

## 1. Problem Statement Recap

When you try to convert a Qwen3.5 model (e.g. `Qwen/Qwen3.5-2B`) with
rkllm-toolkit v1.2.3 you see either:

```
# Scenario A — using AutoModelForCausalLM directly:
ModuleNotFoundError: No module named 'fla'

# Scenario B — even with fla installed:
RuntimeError: No CUDA GPUs are available   (triton requires CUDA)

# Scenario C — using load_huggingface:
WARNING: Not found the weight of model.layers.0.self_attn.q_proj.weight
ERROR: The size of tensor a (2048) must match the size of tensor b (4096)
```

All three errors share the same underlying origin: a mismatch between the
Qwen3.5 architecture and the assumptions baked into transformers + rkllm-toolkit.

---

## 2. Layer 1 — The Transformers Library Problem

### 2a. `qwen3_5` is not in transformers

The rkllm-toolkit standard environment uses `transformers == 4.55.2`.  
You can verify this yourself:

```python
from transformers import CONFIG_MAPPING
print([k for k in CONFIG_MAPPING if 'qwen3' in k])
# → ['qwen3', 'qwen3_moe']
# 'qwen3_5' is MISSING
```

Because `qwen3_5` is not registered, `AutoConfig.from_pretrained('Qwen/Qwen3.5-2B')`
will raise an error unless you add `trust_remote_code=True`.

**Why isn't it in transformers?**  
Qwen3.5's GatedDeltaNet layers depend on the Flash Linear Attention (`fla`)
package for their CUDA kernels.  The transformers maintainers do not merge model
implementations that have hard external kernel dependencies, so Qwen3.5 remains
a `trust_remote_code` model.

### 2b. `trust_remote_code=True` triggers the `fla` import

When you load with `trust_remote_code=True`, Python executes the
`modeling_qwen3_5.py` file bundled **inside the checkpoint directory** (or
downloaded from HuggingFace Hub).  The top of that file contains:

```python
# modeling_qwen3_5.py  (from the official Qwen3.5 checkpoint)
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
```

This import runs **at module load time** — even before any tensors are
created.  If `fla` is not installed, Python fails immediately.

### 2c. `fla` requires `triton ≥ 3.0` which requires CUDA

The Flash Linear Attention (`fla`) library implements its recurrence kernels in
[Triton](https://triton-lang.org/), a Python-embedded CUDA kernel DSL.
Triton's JIT compiler requires:

- A **NVIDIA GPU with CUDA support**
- CUDA toolkit installed
- A compatible driver

This means `fla` (and therefore loading Qwen3.5 through transformers or
trust_remote_code) **cannot work on**:

| Environment | Can install `fla`? |
|---|---|
| NVIDIA GPU + CUDA | ✅ Yes |
| CPU-only machine | ❌ No |
| AMD GPU (ROCm) | ⚠️ Partial (HIP port) |
| Apple Silicon | ❌ No (no CUDA) |
| ARM Linux (Rockchip host for conversion) | ❌ No |

This is precisely what the official developer means by "**the problem is the
Transformers Python library**": the `trust_remote_code` path that
transformers falls back to has a hard CUDA dependency that breaks in most
conversion environments.

### 2d. Summary of the Transformers chain

```
load_huggingface('/path/to/Qwen3.5-2B')
        │
        ▼
AutoConfig.from_pretrained(...)    ← model_type "qwen3_5" not in registry
        │                             → needs trust_remote_code=True
        ▼
Python imports modeling_qwen3_5.py ← from fla.ops.gated_delta_rule import ...
        │                             → fla requires triton
        ▼
triton.runtime.JITFunction         ← needs CUDA GPU
        │
        ▼
❌ ModuleNotFoundError / RuntimeError
```

---

## 3. Layer 2 — The rkllm-runtime Problem

Even if you solve Layer 1 (by providing a custom model that doesn't import
`fla`), there is still a second problem: the **rkllm-runtime has no kernel for
the GatedDeltaNet recurrence**.

When you convert a model with `llm.build(...)`, rkllm-toolkit traces the
model's operations and compiles them to NPU instructions.  For each layer type
it needs a corresponding **NPU kernel** (a compiled routine for the hardware
accelerator).

The rkllm-runtime (v1.2.3) has kernels for:

| Layer type | Kernel |
|---|---|
| Standard attention (MHA, GQA, MQA) | ✅ |
| RWKV7 recurrence (WKV7) | ✅ (added in v1.2.1) |
| SwiGLU MLP | ✅ |
| GatedDeltaNet recurrence | ❌ **missing** |

**Without the GatedDeltaNet kernel**, the toolkit cannot compile a true
Qwen3.5 model.  It can only approximate DeltaNet layers as standard attention
(which is what the adapter in this directory does).

---

## 4. RWKV7 vs GatedDeltaNet — Why One Works and the Other Doesn't

The user is **correct** that RWKV7 is an RNN with linear attention, and that
the framework supports it.  Here is the full comparison:

| Property | RWKV7 | Qwen3.5 GatedDeltaNet |
|---|---|---|
| Architecture family | RNN / Linear Recurrence | RNN / Linear Recurrence |
| Specific formula | WKV7 (see §5) | Delta Rule + Gate (see §5) |
| In transformers registry | `rwkv` (classic), not `rwkv7` | not registered at all |
| External dependency | `rwkv-fla` (pure Python+PyTorch) | `fla` (requires triton/CUDA) |
| CPU fallback exists | ✅ Yes — `rwkv_linear_attention_cpu` in transformers | ❌ No — fla has no CPU path |
| rkllm-runtime kernel | ✅ Added in v1.2.1 | ❌ Not yet added |
| Community can convert | ✅ Yes (Rockchip provided) | ⚠️ Approximation only (this repo) |

### Why RWKV7 works

1. **`rwkv-fla` is a standalone package** — it does NOT require triton for the
   weight-conversion step.  On the conversion host (which may be CPU-only), it
   provides pure-PyTorch implementations of the WKV7 operations that are only
   used to verify or calibrate, not to execute CUDA kernels.

2. **Rockchip added an explicit WKV7 NPU kernel** to rkllm-runtime v1.2.1.  
   This was a deliberate engineering effort to support the RWKV7 architecture.

3. **`rwkv7` is recognized by the toolkit** — rkllm-toolkit v1.2.1 knows how to
   map RWKV7 weight paths to its internal representation.

### Why GatedDeltaNet does NOT work (yet)

1. **`fla` requires triton/CUDA** — the conversion step itself cannot run
   without a CUDA GPU.  This blocks even the first step (loading the model).

2. **No GatedDeltaNet NPU kernel** in rkllm-runtime — Rockchip would need to
   implement `fused_chunk_gated_delta_rule` on the NPU, analogous to `WKV7`.

3. **`qwen3_5` is not in the transformers registry** — rkllm-toolkit uses
   transformers to detect model types; this detection fails silently.

### The conceptual similarity (and why it matters)

Both RWKV7 and GatedDeltaNet are instances of the same abstract pattern:

```
State_{t} = Decay(t) ⊙ State_{t-1}  +  Update(t)
Output_{t} = Readout(State_{t}, Query_{t})
```

The RWKV7 ("WKV7") formula and the GatedDeltaNet formula are two concrete
instantiations of this pattern with different decay and update functions.  
**The NPU *can* execute the GatedDeltaNet recurrence** — it would require an
NPU kernel analogous to the existing WKV7 kernel.  The mathematical operations
involved (matrix-vector multiplications, element-wise scaling, additions) are all
operations that the NPU hardware supports.

---

## 5. The Mathematics of Linear Recurrent Attention

Understanding the math is essential for contributing code.  This section gives
you the formulas so you can write your own implementation.

### 5a. RWKV7 — WKV7 recurrence (for reference)

RWKV7 uses the following recurrence per head (simplified):

```
Inputs per token t:
  r_t  ∈ R^d   receptance (query-like)
  w_t  ∈ R^d   time-decay (data-dependent, per head)
  k_t  ∈ R^d   key
  v_t  ∈ R^d   value
  a_t  ∈ R^d   "a" gate (absorption)
  g_t  ∈ R^d   output gate

State:
  s_t  ∈ R^{d×d}   (matrix per head, d²  floats per head)

Recurrence:
  kv_t = v_t ⊗ k_t                  (outer product, d×d)
  s_t  = diag(w_t) · s_{t-1}
        + kv_t
        - diag(a_t · (s_{t-1} · k_t)) · s_{t-1}   ← "correction" term

Output:
  o_t  = s_t · r_t
  y_t  = o_t * g_t                   (element-wise gating)
```

The key characteristic: `w_t` is **data-dependent** per-token decay,
and there is a correction term involving `a_t` that makes it distinct from
simpler linear attention models.

### 5b. GatedDeltaNet — delta-rule recurrence

This is the algorithm Qwen3.5's DeltaNet layers use.  Understanding this
is the starting point for writing your own implementation.

```
Inputs per token t (per head, head dimension d_k for keys, d_v for values):
  q_t  ∈ R^{d_k}      query    (from q_proj + q_norm)
  k_t  ∈ R^{d_k}      key      (from k_proj + k_norm, L2-normalized to unit sphere)
  v_t  ∈ R^{d_v}      value    (from v_proj)
  β_t  ∈ R            beta gate (from b_proj, scalar per head via sigmoid)
  g_t  ∈ R^{d_v}      decay gate (from g_proj, per-dim per head via sigmoid)

State:
  S_t  ∈ R^{d_k × d_v}     (matrix per head — the "memory")

Step-by-step recurrence:

  1. Read from memory using current key:
       α_t = S_{t-1}ᵀ · k_t        ∈ R^{d_v}

  2. Compute the error (delta):
       δ_t = v_t - α_t              ∈ R^{d_v}

  3. Update the memory with the delta rule:
       S_t = g_t ⊙ S_{t-1}  +  β_t · k_t ⊗ δ_t   ∈ R^{d_k × d_v}
                    ↑                    ↑
              forgetting            learning

     Expanded form (equivalent, see below):
       S_t = g_t ⊙ S_{t-1} · (I - β_t · k_t kᵀ_t)  +  β_t · g_t ⊙ v_t ⊗ k_t

  4. Read output from updated memory:
       o_t = S_t · q_t              ∈ R^{d_v}

  5. Apply output gate (a_proj):
       y_t = o_t * sigmoid(a_proj(x_t))   ∈ R^{d_v}

Full layer output:
  y = concat_heads(y_t) → linear → residual
```

> **Note on step 3 equivalence:**  
> The two expanded forms are mathematically equivalent when `g_t` is a scalar.
> When `g_t` is per-dimension (vector), the second form is the correct one.  
> The FLA library uses a chunked parallel scan algorithm for step 3 to avoid
> the O(seq²) complexity of the sequential loop.

#### Why approximation loses quality

The adapter in `modeling_qwen35.py` replaces steps 1–4 with standard
scaled-dot-product attention.  What is lost:

- **β_t** (the per-token learning rate) is never applied → all tokens
  are treated equally instead of having adaptive memory update strength.
- **g_t** (the per-dimension decay) is never applied → the memory never
  forgets, accumulating all context equally instead of decaying old information.
- **a_proj** (output gating) is discarded → no information filtering at output.
- The **recurrent state** has O(d_k × d_v) size vs. the O(seq × d) KV cache,
  so long-context quality degrades significantly.

---

## 6. What the Community CAN Do Right Now

These tasks do not require Rockchip changes and can be done entirely in Python:

### Task 1 — Implement the pure-PyTorch DeltaNet recurrence (no fla/triton)

The file `deltanet_pure_pytorch.py` in this directory is a reference
implementation.  It runs on CPU and GPU (via standard PyTorch) without any
`fla` or `triton` dependency.

This implementation is useful for:
- **Verifying** that your weight loading is correct (run the recurrence and
  compare outputs to the original `fla`-based model on a GPU machine).
- **Understanding** the math by reading clean, commented code.
- **Prototyping** an NPU-friendly formulation (see Task 2).

```python
from deltanet_pure_pytorch import gated_deltanet_recurrence

# o has the same shape as v: [batch, seq, heads, d_v]
o = gated_deltanet_recurrence(q, k, v, beta, g)
```

### Task 2 — Reformulate the recurrence for NPU-friendly operations

NPU hardware (like the one in RK3588) efficiently supports:
- Matrix-vector multiplication (GEMV)
- Element-wise addition, multiplication
- Small matrix-matrix multiplication (GEMM)

The DeltaNet recurrence (step 3 in §5b) can be written as a series of GEMV
and element-wise ops:

```
α_t   = Sᵀ k_t          (GEMV: d_k×d_v times d_k → d_v)
δ_t   = v_t - α_t       (vector subtraction)
S_t   = g_t ⊙ S_t       (element-wise scale rows by g_t)  ← new
S_t  += β_t * outer(k_t, δ_t)  (rank-1 update, or two GEMV)
o_t   = S_t q_t          (GEMV: d_k×d_v times d_k → d_v)
```

This is the formulation that would need to be implemented in the rkllm-runtime
kernel (in C/C++ with NPU intrinsics).

### Task 3 — Distillation to standard attention

If you want to deploy Qwen3.5 NOW (without waiting for Rockchip), the best
option is to **distill** the DeltaNet layers into standard attention layers by:

1. Taking the full Qwen3.5 model (on a CUDA machine).
2. Running training (or knowledge distillation) where the DeltaNet layers are
   replaced by GQA layers and trained with KL-divergence loss against the original.
3. The result is a "standard" model that runs on rkllm without any approximation.

This is the highest-quality community path that works today.

### Task 4 — Improve the weight adapter

The existing adapter (`modeling_qwen35.py`) discards `b_proj`, `g_proj`,
and `a_proj`.  Even though the NPU runs attention (not DeltaNet), these
weights encode information about which tokens are important and could be used to
**initialize the attention bias** of the approximated layer.

Open research direction: can `b_proj` weights (β gate, learning rate) be
incorporated as an attention mask or bias to partially recover quality?

### Task 5 — Verify exact weight paths for each Qwen3.5 release

Different checkpoint releases may use different sub-module names for the
DeltaNet layers (`self_attn`, `attn`, or `mixer`).  Open the checkpoint and
verify:

```python
from safetensors import safe_open

with safe_open('/path/to/Qwen3.5-2B/model.safetensors', framework='pt') as f:
    for key in f.keys():
        if 'layers.0.' in key:
            print(key)
```

Update `_map_weights` in `convert_qwen35.py` if the paths differ.

### Task 6 — Calibration dataset for Qwen3.5

Better quantization accuracy comes from a calibration dataset that matches
the model's intended use.  Create a JSON file in the format:

```json
[
  {"input": "Human: …\nAssistant: ", "target": "…"},
  …
]
```

with 128–512 representative prompt/response pairs.

---

## 7. What Only Rockchip Can Do

These tasks require modifying the closed-source rkllm-runtime binary and
rkllm-toolkit:

### Runtime Task A — Add a GatedDeltaNet NPU kernel

This is analogous to the WKV7 kernel added in v1.2.1 for RWKV7.

The kernel must implement the recurrence from §5b efficiently on the NPU:

```c
// Pseudocode for one head at one time step
void gated_deltanet_step(
    float* S,          // [d_k × d_v]  state matrix (in-place update)
    const float* q,    // [d_k]        query
    const float* k,    // [d_k]        key (unit-norm)
    const float* v,    // [d_v]        value
    float beta,        // scalar       learning rate gate
    const float* g,    // [d_v]        decay gate (or [1] if scalar)
    float* o           // [d_v]        output
) {
    // Step 1: read from memory
    float alpha[d_v] = matmul_T(S, k);   // S^T * k
    // Step 2: delta
    float delta[d_v];
    for (int i = 0; i < d_v; i++) delta[i] = v[i] - alpha[i];
    // Step 3: decay state
    for (int i = 0; i < d_v; i++)
        for (int j = 0; j < d_k; j++)
            S[j*d_v + i] *= g[i];
    // Step 4: rank-1 update
    for (int j = 0; j < d_k; j++)
        for (int i = 0; i < d_v; i++)
            S[j*d_v + i] += beta * k[j] * delta[i];
    // Step 5: output
    matmul(o, S, q);   // S * q
}
```

### Runtime Task B — Extend `config_custom.json` schema

Add keys for the extra DeltaNet projections:

```json
{
    "DELTANET_BETA":   "self_attn.b_proj",
    "DELTANET_GATE":   "self_attn.g_proj",
    "DELTANET_OUTPUT_GATE": "self_attn.a_proj"
}
```

### Runtime Task C — O(1) recurrent decode mode

The DeltaNet recurrence's main advantage is **O(1) decode** (constant-time per
token) instead of O(seq) KV cache.  Implementing this would make Qwen3.5
significantly faster and more memory-efficient on edge devices.

---

## 8. A Roadmap for Your Own Contribution

Here is a concrete path for contributing human-made code, from easiest to
hardest:

```
Level 1 (Today, any machine):
  → Read deltanet_pure_pytorch.py  (in this directory)
  → Understand the math, run the tests
  → Verify it matches the fla output on a GPU machine

Level 2 (CPU machine):
  → Use convert_qwen35.py to produce an approximated .rkllm
  → Test it on device, measure quality vs. Qwen3 baseline

Level 3 (GPU machine with CUDA):
  → Run the original Qwen3.5 side-by-side with the adapter
  → Measure the quality gap (perplexity, MMLU score)
  → Experiment with incorporating b_proj/g_proj into the adapter (Task 4)

Level 4 (Research direction):
  → Train a distilled Qwen3.5 with DeltaNet → GQA replacement (Task 3)
  → Benchmark on device

Level 5 (For Rockchip engineers or embedded kernel devs):
  → Implement the GatedDeltaNet kernel in rkllm-runtime
  → This is the only path to correct, full-quality Qwen3.5 on NPU
```

---

## References

- [Flash Linear Attention library (fla)](https://github.com/fla-hub/flash-linear-attention) — source of GatedDeltaNet CUDA kernels  
- [DeltaNet paper: "Linear Transformers Are Secretly Fast Weight Memory Systems"](https://arxiv.org/abs/2102.11174) — original delta rule  
- [GatedDeltaNet: "DeltaNet Revisited"](https://arxiv.org/abs/2412.06464) — adds the forgetting gate  
- [RWKV7 / Eagle paper](https://arxiv.org/abs/2406.06565) — WKV7 recurrence  
- [Triton programming guide](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) — why fla needs CUDA  
- `deltanet_pure_pytorch.py` (this directory) — pure-PyTorch reference implementation  
