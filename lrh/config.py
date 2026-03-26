"""
Configuration for ROME x Linear Representation Hypothesis experiments.

References:
    - Meng et al. (2022). Locating and Editing Factual Associations in GPT. NeurIPS.
    - Park et al. (2023). The Linear Representation Hypothesis and the Geometry
      of Large Language Models. arXiv:2311.03658.
    - Hernandez et al. (2024). Linearity of Relation Decoding in Transformer
      Language Models. ICLR.
    - Geiger et al. (2024). Finding Alignments Between Interpretable Causal
      Variables and Distributed Neural Representations. arXiv:2303.02536.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LRHConfig:
    """Configuration for LRH analysis experiments."""

    # Model
    model_name: str = "gpt2-xl"
    device: str = "cuda"
    seed: int = 42

    # --- Dimensional reference (GPT-2 XL) ---
    # d_model = 1600  (residual stream)
    # d_inner = 6400  (MLP hidden / c_proj input)
    # n_layer = 48
    # ROME targets layer 17, edits c_proj weight (6400, 1600)

    # Layer selection
    probe_layers: List[int] = field(
        default_factory=lambda: [0, 5, 10, 15, 17, 20, 25, 30, 35, 40, 47]
    )
    rome_target_layer: int = 17

    # Probe training
    probe_train_size: int = 500
    probe_test_size: int = 200
    probe_lr: float = 1e-3
    probe_epochs: int = 100
    probe_weight_decay: float = 1e-2
    probe_batch_size: int = 64

    # Concept direction extraction
    n_contrastive_pairs: int = 200
    direction_method: str = "mean_diff"  # "mean_diff", "das", "logistic"

    # Linear Relational Embedding (Hernandez et al. 2024)
    lre_rank: int = 1
    lre_n_samples: int = 500
    lre_subject_layer: int = 8
    lre_object_layer: int = 36
    lre_ridge_alpha: float = 1.0

    # ROME bridge analysis
    n_concept_directions: int = 10

    # Module templates (GPT-2 family)
    residual_module_tmp: str = "transformer.h.{}"
    mlp_module_tmp: str = "transformer.h.{}.mlp"
    mlp_proj_module_tmp: str = "transformer.h.{}.mlp.c_proj"
    attn_module_tmp: str = "transformer.h.{}.attn"
    ln_f_module: str = "transformer.ln_f"
    lm_head_module: str = "transformer.wte"

    # Extraction
    extraction_batch_size: int = 32
    token_strategy: str = "subject_last"

    @classmethod
    def from_json(cls, fpath):
        import json

        with open(fpath, "r") as f:
            data = json.load(f)
        return cls(**data)