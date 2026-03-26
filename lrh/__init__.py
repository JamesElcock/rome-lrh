"""
ROME x Linear Representation Hypothesis (LRH) Analysis Module.

This module provides tools to investigate how Rank-One Model Editing
(ROME) interacts with the Linear Representation Hypothesis — the claim
that neural networks encode high-level concepts as linear directions
in their activation spaces.

Core research questions:
    RQ1: Do ROME edits preserve existing linear representations? (probes.py)
    RQ2: Does ROME's v vector align with concept directions? (concept_directions.py)
    RQ3: How does the rank-1 update decompose in concept subspaces? (rome_lrh_bridge.py)
    RQ4: How does ROME's implicit LRE compare to extracted LREs? (lre.py)
    RQ5: Can linear structure predict ROME edit success? (metrics.py)

Dimensional conventions (GPT-2 XL):
    d_model = 1600  (residual stream, ROME's v vector)
    d_inner = 6400  (MLP key space, ROME's u vector)

References:
    Meng et al. (2022). Locating and Editing Factual Associations in GPT.
    Park et al. (2023). The Linear Representation Hypothesis. arXiv:2311.03658.
    Hernandez et al. (2024). Linearity of Relation Decoding in Transformers. ICLR.
"""

from .config import LRHConfig
from .extraction import ActivationExtractor, extract_layer_activations
from .probes import (
    LinearProbe,
    compute_probe_coherence,
    evaluate_probe,
    train_probe,
    train_probes_across_layers,
)
from .concept_directions import (
    ConceptDirection,
    ConceptDirectionBank,
    das_direction,
    extract_relation_directions,
    logistic_direction,
    mean_difference_direction,
)
from .lre import (
    LinearRelationalEmbedding,
    compare_lre_to_rome,
    extract_lre,
    rome_as_implicit_lre,
)
from .rome_lrh_bridge import (
    ROMELRHAnalysis,
    compute_edit_gate_alignment,
    decompose_v_in_concept_basis,
    extract_rome_edit_vectors,
    full_rome_lrh_analysis,
)
from .metrics import (
    d_prime,
    direction_alignment,
    edit_success_correlation,
    explained_variance_by_directions,
    grassmann_distance,
    probe_coherence_delta,
    projection_magnitude,
    subspace_overlap,
)
from .datasets import (
    ProbeDataset,
    RelationGroupedDataset,
    load_lrh_dataset,
)