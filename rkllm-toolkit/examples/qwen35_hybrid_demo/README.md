# Qwen3.5 (Qwen3-Next) Hybrid Model: RKLLM Support Guide

This guide explains the Qwen3.5 architecture, why it fails with rkllm ≤ 1.2.3, and
how to work toward NPU deployment on RK3588/RK3576/RK3562.

---

## 1. Why Qwen3.5 Fails Today (The Root Cause)

When you try to convert `Qwen/Qwen3.5-2B` with the current toolkit you see:

```
WARNING: Not found the weight of model.layers.0.self_attn.q_proj.weight
WARNING: Not found the weight of model.layers.0.self_attn.k_proj.weight
...
ERROR: The size of tensor a (2048) must match the size of tensor b (4096)
```

There are **two independent blockers**:

| Blocker | Root cause | Fix |
|---------|-----------|-----|
| Weight warnings | Layers 0–2 are **not** attention layers; they are GatedDeltaNet recurrent layers with completely different weight names | Use `transformers >= 4.57.0` + hybrid-aware loader |
| Tensor shape mismatch | In `full_attention` layers the `q_proj` now produces **query + gate concatenated** (2× expected size), not just the query | Same — the loader must know the new attention variant |

Both blockers come from a single fact: **Qwen3.5 is a hybrid architecture** that the
current rkllm binary does not yet recognise.

---

## 2. What Is the Qwen3.5 (Qwen3-Next) Architecture?

Qwen3.5 is registered as `model_type = "qwen3_next"` in its `config.json`.
It is a **hybrid sequence model** that alternates between two kinds of decoder layer:

```
layer 0  → linear_attention  (GatedDeltaNet – recurrent)
layer 1  → linear_attention
layer 2  → linear_attention
layer 3  → full_attention    (standard Transformer attention)
layer 4  → linear_attention
...
```

The default ratio is **3 GatedDeltaNet layers for every 1 Transformer layer**
(for a 48-layer model that gives 36 recurrent + 12 attention layers).

### 2a. The `full_attention` Layer (Transformer)

This is a modified Qwen3 attention block.  The key difference from plain Qwen3 is
that **`q_proj` outputs query AND a gate vector** concatenated together:

```python
# q_proj output shape: (batch, seq_len, num_heads * head_dim * 2)
query, gate = q_proj(hidden_states).chunk(2, dim=-1)
# gate is applied after attention via sigmoid:
output = attention(query, k, v) * sigmoid(gate)
output = o_proj(output)
```

Weight names in this layer type:
- `self_attn.q_proj`  — shape `[num_heads * head_dim * 2, hidden_size]`
- `self_attn.k_proj`  — shape `[num_kv_heads * head_dim, hidden_size]`
- `self_attn.v_proj`  — shape `[num_kv_heads * head_dim, hidden_size]`
- `self_attn.o_proj`  — shape `[hidden_size, num_heads * head_dim]`
- `self_attn.q_norm` and `self_attn.k_norm`  — per-head RMS norms

### 2b. The `linear_attention` Layer (GatedDeltaNet)

GatedDeltaNet is a **linear recurrent attention** layer.  It maintains a hidden
*recurrent state matrix* `S` that is updated token by token:

```
# At each token t:
S_t = S_{t-1} * exp(decay_t) + delta_t * (k_t^T · v_t)
y_t = q_t · S_t
output_t = norm(y_t) ⊙ sigmoid(z_t)
output_t = out_proj(output_t)
```

Weight names in this layer type:
- `linear_attn.in_proj_qkvz`  — projects hidden → [q, k, v, z] combined
- `linear_attn.in_proj_ba`    — projects hidden → [beta, alpha] (decay gates)
- `linear_attn.conv1d`        — short depthwise conv applied to [q, k, v]
- `linear_attn.dt_bias`       — time-step bias (like RWKV7's time_mix)
- `linear_attn.A_log`         — log of decay magnitude A
- `linear_attn.norm`          — gated RMS norm on the value output
- `linear_attn.out_proj`      — projects value dimension → hidden_size

### 2c. Why This Matters for rkllm

The rkllm toolkit tries to read `model.layers.N.self_attn.{q,k,v,o}_proj` for
**every** layer.  For `linear_attention` layers those weights do not exist, hence
the "Not found" warnings.  After all warnings, the toolkit attempts to embed the
layer outputs and hits the shape mismatch on the `q_proj` of the first
`full_attention` layer (because `q_proj` is 2× wider than the toolkit expects).

---

## 3. What Is GatedDeltaNet and How Does It Relate to RWKV7?

RWKV7 and GatedDeltaNet are both members of the **linear recurrent attention**
family.  Both maintain a recurrent key-value state matrix.  Their update rules
are closely related:

| Property | RWKV7 | GatedDeltaNet |
|----------|-------|---------------|
| State update | `S = w*S + a*(k^T)v + k^T*v` | `S = exp(g)*S + delta*(k^T)v` |
| Output | `q*S` (with bonus) | `q*S` |
| Gating | token-wise time-mix | exponential decay `exp(g)` |
| Short conv | ✗ (no) | ✓ (causal Conv1D before QKV) |
| rkllm support | ✅ since v1.2.1 | ❌ not yet |

Because RWKV7 is already supported (the rkllm NPU kernel can run recurrent
state updates), GatedDeltaNet support is a natural extension.  The runtime
already has the primitives needed (element-wise decay, outer-product state
update, matrix-vector multiply).

---

## 4. Dependency Requirements

Install the Qwen3.5-specific requirements before conversion:

```bash
pip install -r requirements_qwen35.txt
```

Key additions over the base `requirements.txt`:

| Package | Version | Reason |
|---------|---------|--------|
| `transformers` | ≥ 4.57.0 | Adds `Qwen3NextForCausalLM` (`qwen3_next` model type) |
| `rwkv-fla` | 0.7.x | Provides `fla.ops.gated_delta_rule` CUDA kernels used by GatedDeltaNet |

> **Note**: `transformers == 4.55.2` (pinned in the base
> `rkllm-toolkit/packages/requirements.txt`) does **not** contain
> the `qwen3_next` model type.  You will get `KeyError: 'qwen3_next'` or a
> `ValueError` when trying to load Qwen3.5 with that version.

---

## 5. Conversion Pipeline (Once Native rkllm Support Lands)

The conversion follows the same two-stage pattern used by other supported models:

```
Qwen3.5 HuggingFace checkpoint
         │
         ▼
[Step A] Load with transformers >= 4.57.0
         │   Verify architecture: model_type = "qwen3_next"
         │   Inspect layer_types: ['linear_attention','linear_attention',
         │                          'linear_attention','full_attention', ...]
         │
         ▼
[Step B] llm.load_huggingface(model=path, ...)
         │   (requires rkllm binary to support "qwen3_next")
         │
         ▼
[Step C] llm.build(do_quantization=True, target_platform='rk3588', ...)
         │   NPU kernel handles:
         │     • full_attention  → standard attention KV-cache kernel
         │     • linear_attention → GatedDeltaNet recurrent kernel
         │
         ▼
[Step D] llm.export_rkllm("qwen3.5_2b_rk3588.rkllm")
```

At runtime (on-device), the model alternates between:
- KV-cache inference for `full_attention` layers (just like Qwen3)
- Recurrent-state inference for `linear_attention` layers (just like RWKV7)

---

## 6. Immediate Workaround: Custom Model Path

While waiting for native `qwen3_next` support in the rkllm binary, you can use
the **custom model** conversion path.  See `modeling_qwen35.py` and
`config_qwen35.json` in this directory.

The custom model wraps Qwen3.5 as follows:

1. **For `full_attention` layers**: standard attention is preserved; the gated Q
   projection is handled by splitting `q_proj` output inside the wrapper.

2. **For `linear_attention` layers**: the GatedDeltaNet recurrent kernel is
   approximated as a causal self-attention using the projected keys, queries and
   values.  **This approximation loses the recurrent state across chunks and will
   degrade model quality**, but it produces a *runnable* rkllm model that
   demonstrates the conversion pipeline.  Think of it as a drop-in placeholder
   until the rkllm team adds native GatedDeltaNet support.

```bash
# Step 1 – install Qwen3.5 dependencies
pip install -r ../../packages/requirements_qwen35.txt

# Step 2 – run the conversion script
python export_rkllm_qwen35.py \
    --path /path/to/Qwen3.5-2B \
    --target-platform rk3588 \
    --num_npu_core 3 \
    --quantized_dtype w8a8

# The script will:
#  (a) Load the model with transformers >= 4.57.0
#  (b) Wrap it in the hybrid custom model
#  (c) Call llm.load_huggingface() with the custom config
#  (d) Export to .rkllm
```

---

## 7. What the rkllm Team Needs to Add for Native Support

For the rkllm binary to support Qwen3.5 natively (without the quality-degrading
workaround), the following additions are needed:

### 7a. New model-type recognition

Add `"qwen3_next"` to the list of supported `model_type` values in the rkllm
toolkit, alongside `"qwen3"`, `"rwkv7"`, etc.

### 7b. Hybrid layer dispatcher

The layer iterator currently assumes all layers are the same type.  It needs to
read `config.layer_types[i]` (a list of strings) and dispatch each layer `i` to
the appropriate kernel:

```python
for i, layer_type in enumerate(config.layer_types):
    if layer_type == "full_attention":
        # use existing Qwen3 attention kernel
        run_full_attention_layer(layer_weights[i], ...)
    elif layer_type == "linear_attention":
        # use new GatedDeltaNet recurrent kernel (similar to RWKV7 kernel)
        run_gated_deltanet_layer(layer_weights[i], ...)
```

### 7c. GatedDeltaNet NPU kernel

Implement the recurrent update rule on the NPU.  The pseudocode for the
decode (single-token) step is:

```
# Inputs per token t
mixed_qkvz = in_proj_qkvz(x)           # [batch, key_dim*2 + val_dim*2]
mixed_ba   = in_proj_ba(x)             # [batch, num_heads*2]

# Conv1D state update (short-range context)
conv_state = update_conv(conv_state, mixed_qkvz[:, :conv_dim])
q, k, v, z, beta, alpha = split_and_reshape(mixed_qkvz, mixed_ba)

# Exponential decay
decay = exp(-softplus(A_log) * exp(dt_bias + delta_t))

# Recurrent state update (Delta Rule)
k_normalised = k / (||k|| + eps)
recurrent_state = recurrent_state * decay + k_normalised^T · (v - recurrent_state · k_normalised)

# Output
y = q · recurrent_state            # [batch, num_heads, head_v_dim]
y = RMSNorm_gated(y, z)           # gated normalization
output = out_proj(y.flatten(-2))   # [batch, hidden_size]
```

This is structurally very similar to the existing RWKV7 kernel.  The main
differences are the **conv1d state**, the **delta-rule correction** `(v - S·k)`,
and the **gated RMS norm**.

### 7d. KV-cache extension

The `Qwen3NextDynamicCache` must be extended in the runtime to hold both:
- Standard key/value tensors for `full_attention` layers
- `conv_state` and `recurrent_state` tensors for `linear_attention` layers

---

## 8. Frequently Asked Questions

**Q: Can I just use the Qwen3 (non-hybrid) model instead?**

Yes.  `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, etc. are pure Transformer models
(`model_type = "qwen3"`) and are **already supported** in rkllm v1.2.1+.
They are larger for the same parameter count because they lack the efficiency
gains of the recurrent layers.

**Q: Is Qwen3.5 the same as Qwen3-Next?**

In HuggingFace transformers (≥ 4.57.0) the architecture is registered as
`qwen3_next`.  Alibaba/Qwen markets the released checkpoints as "Qwen3.5".
They are the same model family.

**Q: Will performance on the NPU be good even with hybrid layers?**

Yes — possibly better than pure Transformer models of the same size.  The
recurrent layers have **O(1) memory per token** at decode time (no growing
KV-cache), which means:
- Lower NPU SRAM pressure
- Faster decode for long sequences
- Smaller total memory footprint

**Q: Is the `rwkv-fla` package the same as `flash-linear-attention`?**

`rwkv-fla` is a fork of `flash-linear-attention` (fla) maintained for use
with RWKV models.  Both provide `fla.ops.gated_delta_rule` which is required
by `Qwen3NextGatedDeltaNet`.  The `rwkv-fla` version is already in
`requirements_rwkv7.txt`; we reuse it in `requirements_qwen35.txt`.
