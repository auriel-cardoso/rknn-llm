"""
convert_qwen35.py
=================
End-to-end conversion script for Qwen3.5 models to the rkllm format.

Usage
-----
    python convert_qwen35.py \
        --model_path /path/to/Qwen3.5-2B \
        --target_platform rk3588 \
        --quantized_dtype w4a16 \
        --output qwen3.5-2b-w4a16-rk3588.rkllm

Requirements
------------
Install the packages from:
    rkllm-toolkit/packages/requirements_qwen35.txt

Then install rkllm-toolkit itself:
    pip install rkllm_toolkit-1.2.3-cp310-cp310-linux_x86_64.whl  # adjust for your Python version

Background
----------
Qwen3.5 uses a *hybrid* architecture that interleaves standard Grouped-Query
Attention (GQA) layers with Gated DeltaNet linear-recurrent layers.  The stock
rkllm-toolkit cannot load this model because it searches for attention weights
(q_proj, k_proj, …) at paths that only exist in the GQA layers, not in the
DeltaNet ones.  This script works around that by:

  1. Loading the original Qwen3.5 checkpoint to discover the exact
     configuration (hidden_size, which layers are DeltaNet, etc.).
  2. Building a *custom* :class:`Qwen35ForCausalLM` model that presents a
     unified ``self_attn.*`` interface for every layer (GQA or DeltaNet).
  3. Copying / mapping the original weights into the custom model so nothing
     is lost except the extra DeltaNet gating projections (b_proj, g_proj,
     a_proj) which the rkllm config_custom.json has no slots for.
  4. Saving the adapted model to a temporary directory and calling
     ``llm.load_huggingface`` with the ``custom_config`` parameter.

Quality note
------------
DeltaNet layers are converted to standard attention using the same q/k/v/o
projections.  This is a **best-effort approximation**; the model runs on the
NPU but DeltaNet's linear-recurrence advantage over long contexts is lost.
Full support requires Rockchip to add a GatedDeltaNet kernel to rkllm-runtime.
"""

import argparse
import json
import os
import shutil
import tempfile

import torch
from transformers import AutoConfig, AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Convert Qwen3.5 → rkllm format")
    p.add_argument("--model_path", required=True, help="Path to the Qwen3.5 HuggingFace checkpoint directory")
    p.add_argument("--target_platform", default="rk3588", choices=["rk3562", "rk3576", "rk3588", "rv1126b"])
    p.add_argument("--quantized_dtype", default="w4a16", choices=["w4a16", "w8a8", "w4a16_gx", "w8a8_gx"])
    p.add_argument("--quantized_algorithm", default="normal", choices=["normal", "grq"])
    p.add_argument("--num_npu_core", type=int, default=3)
    p.add_argument("--optimization_level", type=int, default=1)
    p.add_argument("--dataset", default=None, help="Path to calibration dataset JSON (optional)")
    p.add_argument("--max_context", type=int, default=4096)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--output", default=None, help="Output .rkllm file path (auto-named if not given)")
    p.add_argument("--tmp_dir", default=None, help="Temp dir for adapted model (auto if not given)")
    p.add_argument("--keep_tmp", action="store_true", help="Keep the temporary adapted model directory")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: discover Qwen3.5 architecture
# ──────────────────────────────────────────────────────────────────────────────

def _load_raw_config(model_path: str) -> dict:
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)


def _discover_attn_layer_indices(model_path: str, raw_cfg: dict) -> list:
    """
    Determine which layer indices are standard GQA attention layers.

    Strategy (in order of preference):
      1. ``attn_layer_indices`` key in config.json
      2. ``layer_types`` list in config.json  →  ["attention", "deltanet", …]
      3. Inspect the model's state_dict for the presence of
         ``model.layers.N.self_attn.q_proj.weight`` (attention layers have
         it; DeltaNet layers either don't or use a different sub-path).
    """
    if "attn_layer_indices" in raw_cfg:
        return list(raw_cfg["attn_layer_indices"])

    if "layer_types" in raw_cfg:
        return [i for i, t in enumerate(raw_cfg["layer_types"]) if t == "attention"]

    # Fallback: load the state_dict (weights_only) and inspect.
    print("[INFO] Probing state_dict to identify attention layers …")
    from safetensors.torch import load_file as safe_load

    # Try safetensors first.
    shard_files = sorted(
        f for f in os.listdir(model_path)
        if f.endswith(".safetensors") and "non_lora" not in f
    )
    if not shard_files:
        shard_files = sorted(f for f in os.listdir(model_path) if f.endswith(".bin"))
        loader = torch.load
    else:
        loader = safe_load

    attn_indices = []
    num_layers = raw_cfg.get("num_hidden_layers", 0)
    for shard in shard_files:
        shard_path = os.path.join(model_path, shard)
        try:
            sd = loader(shard_path, device="cpu")
        except Exception as e:
            print(f"[WARN] Could not load {shard}: {e}")
            continue
        for layer_idx in range(num_layers):
            # Attention layers have self_attn.q_proj under the default path.
            key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            if key in sd and layer_idx not in attn_indices:
                attn_indices.append(layer_idx)

    print(f"[INFO] Detected {len(attn_indices)} standard-attention layers: {sorted(attn_indices)}")
    return sorted(attn_indices)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: load original weights and map to our custom model
# ──────────────────────────────────────────────────────────────────────────────

def _load_full_state_dict(model_path: str) -> dict:
    """Load all weights from a checkpoint directory into one flat dict."""
    import glob
    sds = {}
    st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if st_files:
        from safetensors.torch import load_file as safe_load
        for f in st_files:
            sds.update(safe_load(f, device="cpu"))
        return sds

    bin_files = sorted(glob.glob(os.path.join(model_path, "*.bin")))
    for f in bin_files:
        # weights_only=True is available from PyTorch ≥ 1.13 and is required
        # by requirements_qwen35.txt (torch==2.6.0).
        sds.update(torch.load(f, map_location="cpu", weights_only=True))
    return sds


def _build_custom_model(raw_cfg: dict, attn_indices: list) -> "Qwen35ForCausalLM":
    """Instantiate the custom Qwen35ForCausalLM with the given config."""
    # Local import so the script can also be run from the repository root.
    try:
        from modeling_qwen35 import Qwen35ForCausalLM
        from configuration_qwen35 import Qwen35Config
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from modeling_qwen35 import Qwen35ForCausalLM
        from configuration_qwen35 import Qwen35Config

    config = Qwen35Config.from_qwen35_config_dict({**raw_cfg, "attn_layer_indices": attn_indices})
    model = Qwen35ForCausalLM(config)
    return model, config


def _map_weights(model, full_sd: dict, raw_cfg: dict, attn_indices: list):
    """
    Copy weights from the original Qwen3.5 state_dict into *model*.

    Standard weight names (embed_tokens, lm_head, norms, MLP projections) are
    loaded directly.  Per-layer attention / DeltaNet weights are routed through
    the layer's load_deltanet_weights() helper.
    """
    num_layers = raw_cfg.get("num_hidden_layers", 0)
    model_sd = model.state_dict()
    new_sd = {}

    # Bulk copy of weights that have identical paths in both models.
    passthrough_keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for k in passthrough_keys:
        if k in full_sd:
            new_sd[k] = full_sd[k]
        elif k in model_sd:
            new_sd[k] = model_sd[k]

    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}."
        is_attn = layer_idx in attn_indices

        # Layer norms and MLP are always the same.
        for sub in ("input_layernorm.weight", "post_attention_layernorm.weight",
                    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
            key = prefix + sub
            if key in full_sd:
                new_sd[key] = full_sd[key]

        # Attention weights.
        if is_attn:
            # Standard GQA: weights live at self_attn.*
            for sub in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                        "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                key = prefix + sub
                if key in full_sd:
                    new_sd[key] = full_sd[key]
        else:
            # DeltaNet: weights may live at self_attn.*, attn.*, or mixer.*
            # Find whichever sub-prefix exists and map q/k/v/o/q_norm/k_norm.
            found_prefix = None
            for sub in ("self_attn", "attn", "mixer"):
                probe = prefix + sub + ".q_proj.weight"
                if probe in full_sd:
                    found_prefix = prefix + sub + "."
                    break

            if found_prefix is not None:
                mapping = {
                    "q_proj.weight": "q_proj.weight",
                    "k_proj.weight": "k_proj.weight",
                    "v_proj.weight": "v_proj.weight",
                    "o_proj.weight": "o_proj.weight",
                    "q_norm.weight": "q_norm.weight",
                    "k_norm.weight": "k_norm.weight",
                }
                for src_sub, tgt_sub in mapping.items():
                    src_key = found_prefix + src_sub
                    tgt_key = prefix + "self_attn." + tgt_sub
                    if src_key in full_sd:
                        src_tensor = full_sd[src_key]
                        tgt_shape = model_sd.get(tgt_key, torch.empty(0)).shape
                        if src_tensor.shape == tgt_shape:
                            new_sd[tgt_key] = src_tensor
                        else:
                            # Pad / truncate to match attention dimensions.
                            from modeling_qwen35 import _align_weight
                            new_sd[tgt_key] = _align_weight(src_tensor, tgt_shape)
            else:
                print(f"[WARN] Layer {layer_idx}: no DeltaNet weights found – "
                      f"keeping random initialisation.")

    missing = [k for k in model_sd if k not in new_sd]
    if missing:
        print(f"[WARN] {len(missing)} weight(s) not found in original checkpoint "
              f"and will keep random values (first 5: {missing[:5]})")

    # Fill any missing keys from model's random init.
    for k in model_sd:
        if k not in new_sd:
            new_sd[k] = model_sd[k]

    model.load_state_dict(new_sd, strict=True)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: save adapted model so rkllm-toolkit can load it
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _save_adapted_model(model, config, original_model_path: str, tmp_dir: str):
    """
    Save the adapted model to *tmp_dir* in a format that rkllm-toolkit can read.

    The directory will contain:
      - adapted model weights (model.safetensors)
      - config.json with auto_map pointing to our custom classes
      - modeling_qwen35.py  (copied from this example directory)
      - configuration_qwen35.py  (copied from this example directory)
      - tokenizer files  (copied from the original model directory)
      - config_custom.json  (the rkllm weight-mapping file)
    """
    os.makedirs(tmp_dir, exist_ok=True)

    # Save weights.
    try:
        from safetensors.torch import save_file as safe_save
        safe_save(model.state_dict(), os.path.join(tmp_dir, "model.safetensors"))
    except ImportError:
        torch.save(model.state_dict(), os.path.join(tmp_dir, "pytorch_model.bin"))

    # Build a config.json with auto_map.
    cfg_dict = {
        "architectures": ["Qwen35ForCausalLM"],
        "model_type": "qwen3_5",
        "auto_map": {
            "AutoConfig": "configuration_qwen35.Qwen35Config",
            "AutoModel": "modeling_qwen35.Qwen35Model",
            "AutoModelForCausalLM": "modeling_qwen35.Qwen35ForCausalLM",
        },
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "attn_layer_indices": config.attn_layer_indices,
        "expand_k": config.expand_k,
        "expand_v": config.expand_v,
        "use_gate": config.use_gate,
        "use_beta": config.use_beta,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "max_position_embeddings": config.max_position_embeddings,
        "vocab_size": config.vocab_size,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
    }
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Copy custom modeling files.
    for fname in ("modeling_qwen35.py", "configuration_qwen35.py", "config_custom.json"):
        src = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp_dir, fname))

    # Copy tokenizer files.
    tok_files = [
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "special_tokens_map.json", "generation_config.json",
    ]
    for fname in tok_files:
        src = os.path.join(original_model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp_dir, fname))

    print(f"[INFO] Adapted model saved to: {tmp_dir}")
    return tmp_dir


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: convert with rkllm-toolkit
# ──────────────────────────────────────────────────────────────────────────────

def _convert(adapted_dir: str, args):
    from rkllm.api import RKLLM

    llm = RKLLM()

    custom_config_path = os.path.join(adapted_dir, "config_custom.json")

    print("[INFO] Loading adapted model into rkllm-toolkit …")
    ret = llm.load_huggingface(
        model=adapted_dir,
        model_lora=None,
        device=args.device,
        dtype=args.dtype,
        custom_config=custom_config_path,
        load_weight=True,
    )
    if ret != 0:
        raise RuntimeError(f"load_huggingface failed with code {ret}")

    print("[INFO] Building quantised model …")
    ret = llm.build(
        do_quantization=True,
        optimization_level=args.optimization_level,
        quantized_dtype=args.quantized_dtype,
        quantized_algorithm=args.quantized_algorithm,
        target_platform=args.target_platform,
        num_npu_core=args.num_npu_core,
        extra_qparams=None,
        dataset=args.dataset,
        max_context=args.max_context,
    )
    if ret != 0:
        raise RuntimeError(f"build failed with code {ret}")

    out_path = args.output or (
        f"qwen3.5_{args.quantized_dtype}_{args.target_platform}.rkllm"
    )
    print(f"[INFO] Exporting to {out_path} …")
    ret = llm.export_rkllm(out_path)
    if ret != 0:
        raise RuntimeError(f"export_rkllm failed with code {ret}")

    print(f"[OK] Conversion complete → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    print("=" * 60)
    print("  Qwen3.5 → rkllm converter")
    print("  (GatedDeltaNet adaptation)")
    print("=" * 60)

    model_path = os.path.expanduser(args.model_path)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # 1. Load raw config and discover architecture.
    raw_cfg = _load_raw_config(model_path)
    attn_indices = _discover_attn_layer_indices(model_path, raw_cfg)

    print(f"[INFO] Model:       {model_path}")
    print(f"[INFO] Layers:      {raw_cfg.get('num_hidden_layers', '?')}")
    print(f"[INFO] Attn layers: {attn_indices or '(all – fallback mode)'}")

    # 2. Build custom model and map weights.
    print("[INFO] Building custom model …")
    model, config = _build_custom_model(raw_cfg, attn_indices)

    print("[INFO] Loading original weights …")
    full_sd = _load_full_state_dict(model_path)

    print("[INFO] Mapping weights …")
    model = _map_weights(model, full_sd, raw_cfg, attn_indices)
    del full_sd  # free memory

    # 3. Save adapted model.
    tmp_dir = args.tmp_dir or tempfile.mkdtemp(prefix="qwen35_adapted_")
    _save_adapted_model(model, config, model_path, tmp_dir)
    del model  # free memory

    # 4. Convert.
    try:
        _convert(tmp_dir, args)
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"[INFO] Temporary directory removed: {tmp_dir}")
        else:
            print(f"[INFO] Temporary directory kept: {tmp_dir}")


if __name__ == "__main__":
    main()
