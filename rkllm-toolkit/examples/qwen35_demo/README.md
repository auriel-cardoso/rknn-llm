# Qwen3.5 → rkllm Conversion Guide

This directory provides a **community-contributed custom-model adapter** that
enables converting Qwen3.5 models (e.g. `Qwen/Qwen3.5-2B`) to the `.rkllm`
format for deployment on Rockchip NPU boards (RK3588, RK3576, RK3562, RV1126B).

---

## Table of Contents

1. [Root Cause of Conversion Failure](#root-cause)
2. [Why GatedDeltaNet Matters](#why-gated-deltanet-matters)
3. [Architecture Overview](#architecture-overview)
4. [What This Adapter Does](#what-this-adapter-does)
5. [Known Limitations](#known-limitations)
6. [Prerequisites](#prerequisites)
7. [Quick Start](#quick-start)
8. [Manual Step-by-Step Conversion](#manual-step-by-step-conversion)
9. [Subtasks for Full Community Support](#subtasks-for-full-community-support)
10. [File Reference](#file-reference)

---

## Root Cause of Conversion Failure <a name="root-cause"></a>

When running `llm.load_huggingface(model='/path/to/Qwen3.5-2B')` with the
stock rkllm-toolkit v1.2.3 you will see:

```
WARNING: Not found the weight of model.layers.0.self_attn.q_proj.weight
WARNING: Not found the weight of model.layers.0.self_attn.k_proj.weight
WARNING: Not found the weight of model.layers.0.self_attn.v_proj.weight
WARNING: Not found the weight of model.layers.0.self_attn.o_proj.weight
WARNING: Not found the weight of model.layers.0.self_attn.q_norm.weight
WARNING: Not found the weight of model.layers.0.self_attn.k_norm.weight
...
ERROR: Catch exception when loading model: The size of tensor a (2048)
       must match the size of tensor b (4096) at non-singleton dimension 0
```

**The cause is a fundamental architectural mismatch:**

| Aspect | Standard attention (Qwen3) | Gated DeltaNet (Qwen3.5) |
|---|---|---|
| Module name | `self_attn` | `attn` / `mixer` / `self_attn`\* |
| Core weights | `q_proj`, `k_proj`, `v_proj`, `o_proj` | same, **plus** `b_proj`, `g_proj`, `a_proj` |
| `q_norm` / `k_norm` | ✔ | ✔ (but may use SubRMSNorm) |
| qk_dim per head | `head_dim = hidden_size / num_heads` | may be smaller (expand_k < 1) |
| Computation | Scaled dot-product attention | Delta-rule linear recurrence |
| KV cache | Standard (key/value tensors) | State matrix (no KV cache) |

\* Exact module name varies between checkpoint releases.

Because rkllm-toolkit hardcodes a single expected layout for every decoder
layer, DeltaNet layers cause "weight not found" warnings and a subsequent
tensor-shape mismatch during the build step.

---

## Why GatedDeltaNet Matters <a name="why-gated-deltanet-matters"></a>

Gated DeltaNet is a **linear-recurrent attention** mechanism based on the
*delta rule* (associative-memory updates).  Compared to standard attention:

- **O(1) inference state** instead of O(seq_len) KV cache → ideal for long
  contexts on memory-limited NPU boards.
- **O(seq_len)** prefill (no quadratic attention) → faster for long prompts.
- **Parameter efficiency**: the recurrent state replaces the KV cache without
  adding extra parameters.

These properties make Qwen3.5 particularly attractive for edge deployment on
Rockchip boards, which is why community support matters.

---

## Architecture Overview <a name="architecture-overview"></a>

Qwen3.5 is a **hybrid** model with two interleaved layer types:

```
Embedding
Layer 0:  [GatedDeltaNet]  ← linear recurrence, no KV cache
Layer 1:  [GQA Attention]  ← standard multi-head attention
Layer 2:  [GatedDeltaNet]
Layer 3:  [GQA Attention]
...
Layer N:  [GQA Attention / GatedDeltaNet]
Final norm + LM head
```

Each decoder layer (regardless of type) also has:
- Pre-attention RMSNorm (`input_layernorm`)
- SwiGLU MLP with gate/up/down projections
- Post-attention RMSNorm (`post_attention_layernorm`)

The exact mixing pattern is stored in `config.json` under one of:
- `attn_layer_indices` – list of integer indices for GQA layers, or
- `layer_types` – list of strings `"attention"` / `"deltanet"` for each layer.

---

## What This Adapter Does <a name="what-this-adapter-does"></a>

The adapter consists of three Python files and one JSON configuration:

| File | Role |
|---|---|
| `configuration_qwen35.py` | `Qwen35Config` – reads `attn_layer_indices` / `layer_types` from the checkpoint config |
| `modeling_qwen35.py` | Custom model that wraps **both** layer types under a unified `self_attn.*` interface |
| `config_custom.json` | rkllm weight-name mapping |
| `convert_qwen35.py` | End-to-end conversion script |

### Weight-path unification

For every layer the adapter exposes weights at exactly the same paths:

```
model.layers.N.self_attn.q_proj.weight
model.layers.N.self_attn.k_proj.weight
model.layers.N.self_attn.v_proj.weight
model.layers.N.self_attn.o_proj.weight
model.layers.N.self_attn.q_norm.weight   ← SubRMSNorm
model.layers.N.self_attn.k_norm.weight   ← SubRMSNorm
```

For **GQA layers** these are straight pass-throughs.

For **DeltaNet layers** the converter:
1. Locates the source weights under whichever sub-path the checkpoint uses
   (`self_attn.*`, `attn.*`, or `mixer.*`).
2. Copies q/k/v/o projections directly; pads/truncates if `qk_dim ≠ head_dim`.
3. Copies q_norm / k_norm weight vectors; resizes if necessary.
4. **Discards** the extra DeltaNet projections (`b_proj`, `g_proj`, `a_proj`)
   because `config_custom.json` has no slots for them – see
   [Subtasks](#subtasks-for-full-community-support).

### Forward-pass approximation

At inference time (NPU) the DeltaNet layers run **standard scaled-dot-product
attention** using the remapped q/k/v/o projections.  This is the best possible
approximation without native DeltaNet kernel support in rkllm-runtime.

---

## Known Limitations <a name="known-limitations"></a>

| Limitation | Explanation |
|---|---|
| **Quality degradation** | DeltaNet → attention approximation loses recurrence dynamics. Impact grows with context length. Short-context tasks (< 512 tokens) see minimal loss; long-context tasks degrade significantly. |
| **No recurrent KV cache** | The NPU cannot exploit DeltaNet's O(1) state. KV cache is used as in standard attention, so memory savings are not realised. |
| **b_proj / g_proj / a_proj discarded** | The output gate and beta gate of DeltaNet are not transferred. This accounts for most of the quality gap. |
| **qk_dim mismatch** | When DeltaNet uses a smaller qk_dim, q/k weights are zero-padded. Fine-tuning on the remapped model can recover some accuracy. |
| **Requires Rockchip runtime changes** | Full, correct support requires Rockchip to add a GatedDeltaNet kernel to `rkllm-runtime`. The community can prepare and test the conversion side, but runtime execution of the true DeltaNet recurrence is not possible without that kernel. |

---

## Prerequisites <a name="prerequisites"></a>

1. **Python 3.10–3.12** (Python 3.12 recommended for consistency with RWKV7)
2. **Install dependencies:**
   ```bash
   pip install -r rkllm-toolkit/packages/requirements_qwen35.txt
   ```
3. **Install rkllm-toolkit** (adjust for your Python version):
   ```bash
   pip install rkllm-toolkit/packages/rkllm_toolkit-1.2.3-cp310-cp310-linux_x86_64.whl
   ```
4. **Download Qwen3.5 model:**
   ```bash
   # Example using huggingface-hub CLI
   huggingface-cli download Qwen/Qwen3.5-2B --local-dir /path/to/Qwen3.5-2B
   ```
5. **GPU** (optional but recommended for quantisation calibration):
   - CUDA-capable GPU with ≥ 16 GB VRAM for Qwen3.5-2B
   - CPU-only is possible but slow

---

## Quick Start <a name="quick-start"></a>

```bash
cd rkllm-toolkit/examples/qwen35_demo

python convert_qwen35.py \
    --model_path /path/to/Qwen3.5-2B \
    --target_platform rk3588 \
    --quantized_dtype w4a16 \
    --device cuda \
    --output qwen3.5-2b-w4a16-rk3588.rkllm
```

**With a calibration dataset** (improves quantisation accuracy):

```bash
python convert_qwen35.py \
    --model_path /path/to/Qwen3.5-2B \
    --target_platform rk3588 \
    --quantized_dtype w4a16 \
    --dataset /path/to/calib.json \
    --output qwen3.5-2b-w4a16-rk3588.rkllm
```

The calibration file format is the same as for other rkllm models:

```json
[
  {"input": "Human: 你好！\nAssistant: ", "target": "你好！"},
  {"input": "Human: What is AI?\nAssistant: ", "target": "Artificial intelligence is …"}
]
```

---

## Manual Step-by-Step Conversion <a name="manual-step-by-step-conversion"></a>

If you prefer to drive the process programmatically:

```python
import sys, os
sys.path.insert(0, 'rkllm-toolkit/examples/qwen35_demo')

from convert_qwen35 import (
    _load_raw_config,
    _discover_attn_layer_indices,
    _build_custom_model,
    _load_full_state_dict,
    _map_weights,
    _save_adapted_model,
)
from rkllm.api import RKLLM
import tempfile, shutil

model_path = '/path/to/Qwen3.5-2B'
raw_cfg    = _load_raw_config(model_path)
attn_idx   = _discover_attn_layer_indices(model_path, raw_cfg)
model, cfg = _build_custom_model(raw_cfg, attn_idx)
full_sd    = _load_full_state_dict(model_path)
model      = _map_weights(model, full_sd, raw_cfg, attn_idx)
del full_sd

tmp_dir = tempfile.mkdtemp(prefix='qwen35_adapted_')
_save_adapted_model(model, cfg, model_path, tmp_dir)
del model

llm = RKLLM()
ret = llm.load_huggingface(
    model=tmp_dir,
    custom_config=os.path.join(tmp_dir, 'config_custom.json'),
    device='cpu',
    dtype='float16',
)
assert ret == 0, 'load failed'

ret = llm.build(
    do_quantization=True,
    optimization_level=1,
    quantized_dtype='w4a16',
    target_platform='rk3588',
    num_npu_core=3,
)
assert ret == 0, 'build failed'

ret = llm.export_rkllm('qwen3.5-2b-w4a16-rk3588.rkllm')
assert ret == 0, 'export failed'

shutil.rmtree(tmp_dir)
```

---

## Subtasks for Full Community Support <a name="subtasks-for-full-community-support"></a>

Below is a breakdown of what remains to achieve *correct* (not approximated)
Qwen3.5 support.  Tasks marked ✅ are handled by this PR; tasks marked 🔨
require additional community or Rockchip work.

### ✅ Completed (this PR)

- [x] Root-cause analysis of the weight-loading failure.
- [x] `Qwen35Config` that reads the hybrid layer layout from config.json.
- [x] `Qwen35DecoderLayer` that unifies GQA and DeltaNet weight paths.
- [x] `Qwen35GatedDeltaNetAsAttention` – approximated DeltaNet adapter.
- [x] Weight-mapping helpers (`_map_weights`, `_align_weight`) for shape
      mismatches between DeltaNet qk_dim and attention head_dim.
- [x] `convert_qwen35.py` – end-to-end conversion script.
- [x] `requirements_qwen35.txt` – Python environment spec.

### 🔨 Community Tasks (no Rockchip changes required)

1. **Verify exact weight paths** for each Qwen3.5 release.
   - Open the model in Python, inspect `model.named_modules()` and
     `model.state_dict().keys()` to confirm whether DeltaNet layers use
     `self_attn`, `attn`, or `mixer` as their sub-module name.
   - Update `_discover_attn_layer_indices` and `_map_weights` accordingly.

2. **Distillation / fine-tuning** to recover quality lost from the
   DeltaNet → attention approximation.
   - Train a distilled version of Qwen3.5 where DeltaNet layers are replaced
     with standard attention using KL-divergence matching on the logits.
   - The `Qwen35GQAttention` class in `modeling_qwen35.py` is ready to receive
     the fine-tuned weights.

3. **Add `b_proj` / `g_proj` / `a_proj` to config_custom.json** once
   Rockchip exposes corresponding config keys in a future toolkit version.
   - Suggested new keys: `DELTANET_BETA`, `DELTANET_GATE`, `DELTANET_ABSORB`.

4. **Calibration dataset** specific to Qwen3.5 for better quantisation accuracy.
   - Generate diverse prompt/response pairs covering typical use cases.

5. **Benchmark** approximated vs. original Qwen3.5 on standard LLM benchmarks
   (MMLU, HellaSwag, etc.) to quantify the quality gap.

### 🔨 Rockchip-Required Tasks (cannot be done by community alone)

6. **Add GatedDeltaNet kernel to rkllm-runtime** – this is the most impactful
   change.  The kernel must implement the *fused chunk* recurrence from the FLA
   library (`fused_chunk_delta_rule`) on the NPU or ARM CPU.
   - Reference implementation: [Flash Linear Attention library](https://github.com/fla-hub/flash-linear-attention), specifically `fla.ops.delta_rule`.
   - An NPU-efficient formulation may need reformulation as a sequence of
     matrix-vector operations that map to the NPU's supported op set.

7. **Extend config_custom.json schema** to support extra DeltaNet projections
   (`DELTANET_BETA`, `DELTANET_GATE`, `DELTANET_ABSORB`).

8. **Add O(1) recurrent inference mode** to the rkllm-runtime so that at
   decode time the DeltaNet state matrix is updated rather than a full KV
   cache being maintained.

---

## File Reference <a name="file-reference"></a>

```
rkllm-toolkit/
├── packages/
│   └── requirements_qwen35.txt        ← Python dependencies incl. FLA library
└── examples/
    └── qwen35_demo/
        ├── README.md                  ← this file
        ├── configuration_qwen35.py    ← Qwen35Config (hybrid layout)
        ├── modeling_qwen35.py         ← custom model adapter
        ├── config_custom.json         ← rkllm weight-name mapping
        └── convert_qwen35.py         ← end-to-end conversion script
```
