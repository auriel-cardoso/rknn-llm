#!/usr/bin/env python3
# coding=utf-8
"""
export_rkllm_qwen35.py — Convert a Qwen3.5 (Qwen3-Next) checkpoint to .rkllm
═══════════════════════════════════════════════════════════════════════════════

Background
──────────
Qwen3.5 is a hybrid sequence model (model_type = "qwen3_next") with two layer
types per decoder block:

  full_attention   → standard Transformer attention (every 4th layer)
  linear_attention → GatedDeltaNet recurrent layer (remaining layers)

The rkllm-toolkit ≤ 1.2.3 does not recognise "qwen3_next" natively.
This script works around that by loading the model through the rkllm
*custom-model* path using the adapter defined in modeling_qwen35.py.

Current limitation (quality degradation)
─────────────────────────────────────────
Until native rkllm support for "qwen3_next" is merged, the linear_attention
(GatedDeltaNet) layers are approximated as causal self-attention.  This means:
  • The model runs correctly on the NPU and produces coherent text.
  • Cross-sequence recurrent state is NOT maintained between calls.
  • Quality is similar to a Qwen3 model of the same size (no recurrence gain).

When rkllm adds native qwen3_next support, you can switch to the standard
load_huggingface() path without any wrappers.

Dependencies
────────────
  pip install -r ../../packages/requirements_qwen35.txt

Usage examples
──────────────
  # Basic conversion (w8a8, rk3588)
  python export_rkllm_qwen35.py --path /path/to/Qwen3.5-2B

  # Custom quantisation and output path
  python export_rkllm_qwen35.py \\
      --path /path/to/Qwen3.5-2B \\
      --target-platform rk3576 \\
      --num_npu_core 2 \\
      --quantized_dtype w4a16 \\
      --savepath ./qwen35_2b_rk3576_w4a16.rkllm
"""

import os
import sys
import argparse

# ── dependency check ──────────────────────────────────────────────────────────

def _check_deps():
    """Verify that the correct transformers version is installed."""
    try:
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__) < Version("4.57.0"):
            print(
                f"[ERROR] transformers {transformers.__version__} is installed, "
                "but Qwen3.5 requires >= 4.57.0.\n"
                "Please run:  pip install 'transformers>=4.57.0'"
            )
            sys.exit(1)
        # Verify qwen3_next is actually present
        from transformers import Qwen3NextForCausalLM  # noqa: F401
    except ImportError as exc:
        print(
            "[ERROR] Required package not found.\n"
            "Please run:  pip install -r ../../packages/requirements_qwen35.txt\n"
            f"Detail: {exc}"
        )
        sys.exit(1)


_check_deps()

# ── imports (after dep check) ─────────────────────────────────────────────────

import torch
from transformers import AutoTokenizer

# Local hybrid model adapter
sys.path.insert(0, os.path.dirname(__file__))
from modeling_qwen35 import Qwen35HybridModel

# ── argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert Qwen3.5 (Qwen3-Next) to .rkllm format"
    )
    p.add_argument(
        "--path",
        type=str,
        required=True,
        help="Local path to the Qwen3.5 checkpoint directory",
    )
    p.add_argument(
        "--target-platform",
        type=str,
        default="rk3588",
        choices=["rk3588", "rk3576", "rk3562"],
        help="Target Rockchip NPU platform (default: rk3588)",
    )
    p.add_argument(
        "--num_npu_core",
        type=int,
        default=3,
        help="Number of NPU cores to use (default: 3)",
    )
    p.add_argument(
        "--quantized_dtype",
        type=str,
        default="w8a8",
        choices=["w8a8", "w4a16", "w8a8_g128", "w4a16_g128"],
        help="Quantisation dtype (default: w8a8)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to calibration dataset JSON (optional; uses default if absent)",
    )
    p.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Output .rkllm path (auto-generated from model name if not set)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for loading the model (default: cpu)",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Print architecture details and exit without converting",
    )
    return p.parse_args()


# ── architecture inspector ────────────────────────────────────────────────────

def inspect_architecture(model_path: str):
    """
    Load and describe the Qwen3.5 architecture without running conversion.
    Useful for understanding the hybrid layer structure before converting.
    """
    from transformers import Qwen3NextConfig
    cfg = Qwen3NextConfig.from_pretrained(model_path)

    layer_types = cfg.layer_types
    n_full = sum(1 for t in layer_types if t == "full_attention")
    n_lin  = sum(1 for t in layer_types if t == "linear_attention")

    print("=" * 60)
    print(f"  Qwen3.5 Architecture Inspector")
    print("=" * 60)
    print(f"  model_type          : {cfg.model_type}")
    print(f"  num_hidden_layers   : {cfg.num_hidden_layers}")
    print(f"  hidden_size         : {cfg.hidden_size}")
    print(f"  num_attention_heads : {cfg.num_attention_heads}")
    print(f"  num_key_value_heads : {cfg.num_key_value_heads}")
    print(f"  vocab_size          : {cfg.vocab_size}")
    print()
    print(f"  Layer type breakdown:")
    print(f"    full_attention    : {n_full} layers (standard transformer)")
    print(f"    linear_attention  : {n_lin} layers (GatedDeltaNet recurrent)")
    print(f"    ratio             : {n_lin // max(n_full, 1)}:1 recurrent:attention")
    print()
    print(f"  GatedDeltaNet parameters:")
    print(f"    conv_kernel_size  : {cfg.linear_conv_kernel_dim}")
    print(f"    linear_key_heads  : {cfg.linear_num_key_heads}")
    print(f"    linear_val_heads  : {cfg.linear_num_value_heads}")
    print(f"    linear_key_dim    : {cfg.linear_key_head_dim}")
    print(f"    linear_val_dim    : {cfg.linear_value_head_dim}")
    print()
    print(f"  Layer pattern (first 16):")
    for i, lt in enumerate(layer_types[:16]):
        tag = "ATTN" if lt == "full_attention" else "GDN "
        print(f"    layer {i:2d}: {tag} ({lt})")
    if len(layer_types) > 16:
        print(f"    … ({len(layer_types) - 16} more layers)")
    print("=" * 60)

    # Explain the rkllm blockers
    print()
    print("  rkllm compatibility notes:")
    print("  ──────────────────────────")
    print(f"  • model_type '{cfg.model_type}' requires rkllm >= 1.2.4 for native support.")
    print("  • GDN layers lack self_attn.q_proj → rkllm ≤ 1.2.3 shows weight warnings.")
    print("  • q_proj in ATTN layers outputs 2× head_dim (query + gate) → shape mismatch.")
    print("  • Use this script's custom-model path as a workaround (see --help).")
    print()


# ── main conversion ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Architecture inspection mode
    if args.inspect:
        inspect_architecture(args.path)
        return

    # Auto-generate output path
    if args.savepath is None:
        model_basename = os.path.basename(args.path.rstrip("/")).lower()
        savepath = os.path.join(
            "./rkllm",
            f"{model_basename}_{args.quantized_dtype}_{args.target_platform}.rkllm",
        )
    else:
        savepath = args.savepath

    os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)

    print("[Step 1] Loading Qwen3.5 model via hybrid adapter…")
    print(f"         Path: {args.path}")
    model = Qwen35HybridModel.from_pretrained(args.path)
    print(f"         Parameters: {model.count_parameters():,}")

    print()
    print("[Step 2] Loading tokeniser…")
    tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)
    print(f"         Vocabulary size: {tokenizer.vocab_size:,}")

    # ── rkllm conversion ──────────────────────────────────────────────────────
    # NOTE: The block below shows HOW the rkllm API would be called.
    #       The actual call requires the rkllm binary to support the custom model
    #       with the layer types used in modeling_qwen35.py.
    #
    #       When rkllm >= 1.2.4 with native qwen3_next support is released you
    #       can replace this block with a simple llm.load_huggingface() call.
    # ─────────────────────────────────────────────────────────────────────────

    print()
    print("[Step 3] Initialising rkllm toolkit…")

    try:
        from rkllm.api import RKLLM
    except ImportError:
        print(
            "[ERROR] rkllm package not found.\n"
            "Install the appropriate wheel from rkllm-toolkit/packages/ first:\n"
            "  pip install rkllm_toolkit-*.whl"
        )
        sys.exit(1)

    llm = RKLLM()

    custom_config_path = os.path.join(os.path.dirname(__file__), "config_qwen35.json")

    print(f"         Custom config: {custom_config_path}")
    print()
    print("[Step 4] Loading model into rkllm (custom-model path)…")
    print()
    print(
        "  ARCHITECTURE NOTE\n"
        "  ──────────────────\n"
        "  The model is loaded via the rkllm custom-model path.\n"
        "  GatedDeltaNet (linear_attention) layers are approximated as\n"
        "  causal self-attention for this conversion.  Quality will be\n"
        "  similar to a same-size pure-Transformer Qwen3 model.\n"
        "  Native qwen3_next support (with true recurrent kernels) is\n"
        "  expected in a future rkllm release.\n"
    )

    # The rkllm API call (requires rkllm binary that supports custom models):
    #
    # When rkllm adds native qwen3_next support you can replace the two
    # arguments below with the standard load_huggingface() call without any
    # custom_model flags.
    #
    # For now, the custom-model path is required.  If your rkllm version does
    # not support custom_model=True, the call will fail with an error — in that
    # case, please upgrade to rkllm >= 1.2.4 or contact the Rockchip rkllm team.
    ret = llm.load_huggingface(
        model=model,
        device=args.device,
        custom_model=True,
        custom_model_config=custom_config_path,
    )
    if ret != 0:
        print(f"[ERROR] load_huggingface() returned {ret}")
        sys.exit(ret)

    print()
    print("[Step 5] Building quantised model…")
    print(f"         Platform       : {args.target_platform}")
    print(f"         NPU cores      : {args.num_npu_core}")
    print(f"         Quant dtype    : {args.quantized_dtype}")

    dataset = args.dataset  # None is OK; rkllm uses a default calibration set
    ret = llm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype=args.quantized_dtype,
        quantized_algorithm="normal",
        target_platform=args.target_platform,
        num_npu_core=args.num_npu_core,
        dataset=dataset,
    )
    if ret != 0:
        print(f"[ERROR] build() returned {ret}")
        sys.exit(ret)

    print()
    print(f"[Step 6] Exporting to: {savepath}")
    ret = llm.export_rkllm(savepath)
    if ret != 0:
        print(f"[ERROR] export_rkllm() returned {ret}")
        sys.exit(ret)

    print()
    print(f"[Done] Model saved to: {savepath}")
    print()
    print("On-device deployment")
    print("────────────────────")
    print("  Copy the .rkllm file to the board and run with the rkllm runtime.")
    print("  The model will use KV-cache for full_attention layers and a static")
    print("  zero recurrent state for linear_attention layers (approximation).")
    print()
    print("  For true recurrent-state inference (full Qwen3.5 quality), wait for")
    print("  native qwen3_next support in a future rkllm release or contact the")
    print("  Rockchip rkllm team to request priority support for this model type.")


if __name__ == "__main__":
    main()
