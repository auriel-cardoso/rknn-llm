# coding=utf-8
# Configuration class for Qwen3.5 hybrid architecture (GatedDeltaNet + GQA).
#
# Qwen3.5 is a hybrid language model that interleaves standard Grouped-Query
# Attention (GQA) layers with Gated DeltaNet linear-attention layers.
# This file defines the configuration that captures both layer types so that
# the rkllm-toolkit custom-model interface can load and convert the model.

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen35Config(PretrainedConfig):
    r"""
    Configuration class for the Qwen3.5 hybrid (GatedDeltaNet + GQA) model.

    Qwen3.5 interleaves two fundamentally different layer types:
      * **Standard GQA attention** layers – identical to Qwen3.
      * **Gated DeltaNet** layers – a linear-recurrent layer from the Flash
        Linear Attention (FLA) library.  These layers do *not* have the
        conventional ``self_attn.{q,k,v,o}_proj`` weight paths expected by
        the rkllm-toolkit, which is the root cause of the conversion failure
        reported in airockchip/rknn-llm#472.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    hidden_size : int
        Dimension of the hidden representations.
    intermediate_size : int
        Inner dimension of the SwiGLU MLP.
    num_hidden_layers : int
        Total number of decoder layers (attention + DeltaNet combined).
    num_attention_heads : int
        Number of query heads for all layer types.
    num_key_value_heads : int
        Number of key/value heads (GQA).  Defaults to ``num_attention_heads``
        (MHA) when not specified.
    head_dim : int | None
        Per-head dimension.  Derived from ``hidden_size / num_attention_heads``
        when ``None``.
    attn_layer_indices : list[int] | None
        Zero-based indices of layers that use **standard GQA attention**.
        All other layers are treated as **Gated DeltaNet**.
        When ``None`` every layer is treated as standard attention (full
        fallback mode, useful for debugging).
    expand_k : int
        Key/query expansion ratio for DeltaNet layers (default 1).
    expand_v : int
        Value expansion ratio for DeltaNet layers (default 2).
    use_gate : bool
        Whether DeltaNet layers have an output gate (``g_proj``).
    use_beta : bool
        Whether DeltaNet layers have a per-head beta gate (``b_proj``).
    rms_norm_eps : float
        Epsilon for RMSNorm layers.
    rope_theta : float
        Base for RoPE position embeddings.
    rope_scaling : dict | None
        Optional RoPE scaling configuration.
    max_position_embeddings : int
        Maximum sequence length.
    tie_word_embeddings : bool
        Whether to tie input and output embeddings.
    """

    model_type = "qwen3_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = None,
        attn_layer_indices: list = None,
        expand_k: int = 1,
        expand_v: int = 2,
        use_gate: bool = True,
        use_beta: bool = True,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        rope_scaling: dict = None,
        max_position_embeddings: int = 32768,
        tie_word_embeddings: bool = False,
        pad_token_id: int = None,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads

        # Which layers are standard GQA attention (the rest are GatedDeltaNet).
        # An empty/None list means ALL layers are treated as standard attention
        # (graceful degradation / debug mode).
        self.attn_layer_indices = attn_layer_indices if attn_layer_indices is not None else []

        # GatedDeltaNet-specific hyper-parameters
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_beta = use_beta

        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def is_attention_layer(self, layer_idx: int) -> bool:
        """Return True if *layer_idx* is a standard GQA attention layer."""
        if not self.attn_layer_indices:
            # Fallback: treat every layer as standard attention.
            return True
        return layer_idx in self.attn_layer_indices

    @classmethod
    def from_qwen35_config_dict(cls, raw: dict) -> "Qwen35Config":
        """
        Build a :class:`Qwen35Config` from the raw ``config.json`` dict of a
        Qwen3.5 HuggingFace checkpoint.

        The raw config is expected to contain at minimum:
        - ``hidden_size``, ``num_hidden_layers``, ``num_attention_heads``,
          ``num_key_value_heads``, ``vocab_size``.
        - Optionally ``attn_layer_indices`` (list of int) – if absent the
          helper will try to infer it from ``layer_types`` (list of strings
          "attention" / "deltanet") if that field is present, otherwise it
          falls back to treating every layer as attention.
        """
        kwargs = dict(raw)  # shallow copy

        # Normalise the layer-type field: Qwen3.5 checkpoints may encode this
        # as a list of strings or as a list of indices.
        if "attn_layer_indices" not in kwargs:
            layer_types = kwargs.pop("layer_types", None)
            if layer_types is not None:
                # e.g. ["attention", "deltanet", "deltanet", "attention", ...]
                kwargs["attn_layer_indices"] = [
                    i for i, t in enumerate(layer_types) if t == "attention"
                ]
            else:
                kwargs["attn_layer_indices"] = []

        # Drop fields that the parent PretrainedConfig doesn't know about
        # but that might be present in raw Qwen3/3.5 configs.
        for field in ("architectures", "model_type", "transformers_version",
                      "auto_map", "sliding_window", "use_sliding_window",
                      "use_cache"):
            kwargs.pop(field, None)

        return cls(**kwargs)
