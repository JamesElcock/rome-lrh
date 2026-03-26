"""
Concept direction extraction for the Linear Representation Hypothesis.

Implements several methods for identifying linear directions in activation
space that encode specific concepts:

    1. Mean difference (contrastive centroid subtraction)
    2. DAS (Distributed Alignment Search) via SVD
    3. Logistic probe direction (trained classifier weight)

All methods produce unit vectors in activation space. For residual stream
extraction these are in R^{d_model} (1600 for GPT-2 XL) and are directly
comparable to ROME's v vector. For MLP key space extraction these are in
R^{d_inner} (6400) and are comparable to ROME's u vector.

References:
    - Park et al. (2023). The Linear Representation Hypothesis and the
      Geometry of Large Language Models. arXiv:2311.03658.
    - Geiger et al. (2024). Finding Alignments Between Interpretable Causal
      Variables and Distributed Neural Representations. arXiv:2303.02536.
    - Marks & Tegmark (2024). The Geometry of Truth: Emergent Linear
      Structure in Large Language Model Representations of True/False
      Statements. arXiv:2310.06824.
    - Nanda et al. (2023). Emergent Linear Representations in World Models
      of Self-Supervised Sequence Models. arXiv:2309.00941.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .config import LRHConfig
from .extraction import ActivationExtractor
from .probes import LinearProbe, evaluate_probe, train_probe


@dataclass
class ConceptDirection:
    """
    A concept direction in activation space.

    The direction vector is a unit vector whose inner product with an
    activation reveals the degree to which that concept is present.
    """

    direction: torch.Tensor  # (d,), unit vector
    layer: int
    module_type: str  # "residual", "mlp_key", "mlp_out"
    concept_name: str
    method: str  # "mean_diff", "das", "logistic"
    metadata: Dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.direction.size(0)

    def project(self, activations: torch.Tensor) -> torch.Tensor:
        """Project activations onto this concept direction. Returns scalar per sample."""
        return activations.float() @ self.direction.float().to(activations.device)


def mean_difference_direction(
    extractor: ActivationExtractor,
    positive_prompts: List[str],
    positive_subjects: List[str],
    negative_prompts: List[str],
    negative_subjects: List[str],
    layer: int,
    concept_name: str = "",
    module_type: str = "residual",
    token_strategy: str = "subject_last",
) -> ConceptDirection:
    """
    Compute concept direction as mean(positive) - mean(negative).

    This is the simplest contrastive method. It works well when:
    - The concept is approximately linearly encoded
    - The positive/negative sets differ primarily along one axis

    The resulting direction d satisfies: <d, h_pos> > <d, h_neg> on average.

    Reference:
        Burns et al. (2022). Discovering Latent Knowledge in Language Models
        Without Supervision. arXiv:2212.03827. (CCS uses a related principle.)
    """
    # Extract activations for both sets
    extract_fn = {
        "residual": extractor.extract_residual_stream,
        "mlp_key": extractor.extract_mlp_key,
        "mlp_out": extractor.extract_mlp_output,
    }[module_type]

    pos_acts = extract_fn(positive_prompts, positive_subjects, [layer], token_strategy)[
        layer
    ]
    neg_acts = extract_fn(negative_prompts, negative_subjects, [layer], token_strategy)[
        layer
    ]

    # Compute direction as difference of means
    pos_mean = pos_acts.float().mean(dim=0)
    neg_mean = neg_acts.float().mean(dim=0)
    d = pos_mean - neg_mean
    d = d / (d.norm() + 1e-10)

    return ConceptDirection(
        direction=d.cpu(),
        layer=layer,
        module_type=module_type,
        concept_name=concept_name,
        method="mean_diff",
        metadata={
            "n_positive": len(positive_prompts),
            "n_negative": len(negative_prompts),
            "raw_diff_norm": (pos_mean - neg_mean).norm().item(),
        },
    )


def das_direction(
    extractor: ActivationExtractor,
    paired_prompts: List[Tuple[str, str]],
    paired_subjects: List[Tuple[str, str]],
    layer: int,
    concept_name: str = "",
    module_type: str = "residual",
    token_strategy: str = "subject_last",
    rank: int = 1,
) -> ConceptDirection:
    """
    Distributed Alignment Search (DAS) direction extraction.

    Given n paired examples (h_pos_i, h_neg_i), form the difference matrix
    H_diff of shape (n, d) and take its top right singular vector. This
    finds the direction of maximum variance in the *differences*, which is
    more robust than mean-difference when there's substantial within-class
    variation.

    Formally: d = argmax_{||d||=1} Var[<d, h_pos - h_neg>]

    This is the top right singular vector of H_diff = USV^T.

    Reference:
        Geiger et al. (2024). Finding Alignments Between Interpretable
        Causal Variables and Distributed Neural Representations.
    """
    pos_prompts = [p[0] for p in paired_prompts]
    neg_prompts = [p[1] for p in paired_prompts]
    pos_subjects = [s[0] for s in paired_subjects]
    neg_subjects = [s[1] for s in paired_subjects]

    extract_fn = {
        "residual": extractor.extract_residual_stream,
        "mlp_key": extractor.extract_mlp_key,
        "mlp_out": extractor.extract_mlp_output,
    }[module_type]

    pos_acts = extract_fn(pos_prompts, pos_subjects, [layer], token_strategy)[layer]
    neg_acts = extract_fn(neg_prompts, neg_subjects, [layer], token_strategy)[layer]

    # Difference matrix
    H_diff = (pos_acts - neg_acts).float()  # (n, d)

    # Center the differences (subtract mean difference)
    H_diff = H_diff - H_diff.mean(dim=0, keepdim=True)

    # SVD to find direction of maximum variance in differences
    U, S, Vh = torch.linalg.svd(H_diff, full_matrices=False)
    d = Vh[0]  # Top right singular vector, shape (d,)
    d = d / (d.norm() + 1e-10)

    # Explained variance
    total_var = (S**2).sum().item()
    explained = (S[0] ** 2).item() / (total_var + 1e-10) if total_var > 0 else 0.0

    return ConceptDirection(
        direction=d.cpu(),
        layer=layer,
        module_type=module_type,
        concept_name=concept_name,
        method="das",
        metadata={
            "n_pairs": len(paired_prompts),
            "top_singular_value": S[0].item(),
            "explained_variance_ratio": explained,
            "rank": rank,
        },
    )


def logistic_direction(
    extractor: ActivationExtractor,
    positive_prompts: List[str],
    positive_subjects: List[str],
    negative_prompts: List[str],
    negative_subjects: List[str],
    layer: int,
    config: LRHConfig,
    concept_name: str = "",
    module_type: str = "residual",
    token_strategy: str = "subject_last",
) -> ConceptDirection:
    """
    Extract concept direction from a trained logistic regression probe.

    The direction is the normal to the decision hyperplane (W[1] - W[0]),
    which maximizes the margin between the two classes under log-loss.
    Regularization (weight decay) biases the probe toward minimum-norm
    solutions, producing more interpretable directions.

    This is typically the most informative single-direction extraction
    method, as it optimizes for discriminability.
    """
    extract_fn = {
        "residual": extractor.extract_residual_stream,
        "mlp_key": extractor.extract_mlp_key,
        "mlp_out": extractor.extract_mlp_output,
    }[module_type]

    # Extract all activations
    all_prompts = positive_prompts + negative_prompts
    all_subjects = positive_subjects + negative_subjects
    acts = extract_fn(all_prompts, all_subjects, [layer], token_strategy)[layer]

    labels = torch.cat([
        torch.ones(len(positive_prompts), dtype=torch.long),
        torch.zeros(len(negative_prompts), dtype=torch.long),
    ])

    # Train probe
    input_dim = acts.size(1)
    probe = LinearProbe(input_dim, n_classes=2)
    train_probe(probe, acts, labels, config)
    metrics = evaluate_probe(probe, acts, labels)

    return ConceptDirection(
        direction=probe.direction.cpu(),
        layer=layer,
        module_type=module_type,
        concept_name=concept_name,
        method="logistic",
        metadata={
            "n_positive": len(positive_prompts),
            "n_negative": len(negative_prompts),
            "probe_accuracy": metrics["accuracy"],
            "probe_balanced_accuracy": metrics["balanced_accuracy"],
        },
    )


class ConceptDirectionBank:
    """
    Collection of concept directions, organized by layer and concept name.

    Provides convenience methods for decomposing vectors in the concept basis
    and computing pairwise alignment matrices.
    """

    def __init__(self):
        self._directions: Dict[int, Dict[str, ConceptDirection]] = {}

    def add(self, cd: ConceptDirection):
        if cd.layer not in self._directions:
            self._directions[cd.layer] = {}
        self._directions[cd.layer][cd.concept_name] = cd

    def get(self, layer: int, concept_name: str) -> Optional[ConceptDirection]:
        return self._directions.get(layer, {}).get(concept_name)

    def get_layer(self, layer: int) -> Dict[str, ConceptDirection]:
        return self._directions.get(layer, {})

    def layers(self) -> List[int]:
        return sorted(self._directions.keys())

    def concepts_at_layer(self, layer: int) -> List[str]:
        return sorted(self._directions.get(layer, {}).keys())

    def direction_matrix(self, layer: int) -> Tuple[torch.Tensor, List[str]]:
        """
        Stack all directions at a layer into a matrix.

        Returns:
            (matrix of shape (n_concepts, d), list of concept names)
        """
        dirs = self._directions.get(layer, {})
        names = sorted(dirs.keys())
        if not names:
            return torch.empty(0), []
        mat = torch.stack([dirs[n].direction for n in names], dim=0)
        return mat, names

    def pairwise_alignment(self, layer: int) -> Tuple[torch.Tensor, List[str]]:
        """
        Compute pairwise cosine similarity between all concept directions
        at a given layer.

        Returns:
            (similarity matrix of shape (n, n), list of concept names)
        """
        mat, names = self.direction_matrix(layer)
        if mat.size(0) == 0:
            return torch.empty(0, 0), []
        mat = mat.float()
        norms = mat.norm(dim=1, keepdim=True).clamp(min=1e-10)
        sim = (mat / norms) @ (mat / norms).T
        return sim, names


def extract_relation_directions(
    extractor: ActivationExtractor,
    dataset,  # RelationGroupedDataset
    layers: List[int],
    method: str = "mean_diff",
    module_type: str = "residual",
    config: LRHConfig = None,
    max_relations: int = 20,
) -> ConceptDirectionBank:
    """
    Extract concept directions for multiple relations across multiple layers.

    For each relation, uses the most common target value as the "positive"
    class and all other values as "negative". This gives a direction that
    encodes "is this entity associated with [most common target] for this
    relation?"

    Returns:
        ConceptDirectionBank populated with directions for each
        (relation, layer) combination.
    """
    config = config or LRHConfig()
    bank = ConceptDirectionBank()

    relations = dataset.relation_ids[:max_relations]

    for rid in relations:
        targets = dataset.get_unique_targets(rid)
        if len(targets) < 2:
            continue

        # Use most common target as positive class
        records = dataset.get_relation_records(rid)
        from collections import Counter

        target_counts = Counter(
            r["requested_rewrite"]["target_true"]["str"].strip() for r in records
        )
        top_target = target_counts.most_common(1)[0][0]

        pos, neg = dataset.get_contrastive_pairs(
            rid, top_target, n_pairs=config.n_contrastive_pairs
        )
        if len(pos) < 5 or len(neg) < 5:
            continue

        concept_name = f"{rid}_{top_target}"

        for layer in layers:
            if method == "mean_diff":
                cd = mean_difference_direction(
                    extractor,
                    [p["prompt"] for p in pos],
                    [p["subject"] for p in pos],
                    [p["prompt"] for p in neg],
                    [p["subject"] for p in neg],
                    layer=layer,
                    concept_name=concept_name,
                    module_type=module_type,
                )
            elif method == "logistic":
                cd = logistic_direction(
                    extractor,
                    [p["prompt"] for p in pos],
                    [p["subject"] for p in pos],
                    [p["prompt"] for p in neg],
                    [p["subject"] for p in neg],
                    layer=layer,
                    config=config,
                    concept_name=concept_name,
                    module_type=module_type,
                )
            elif method == "das":
                pairs = dataset.get_paired_contrastive(
                    rid, n_pairs=config.n_contrastive_pairs
                )
                if len(pairs) < 5:
                    continue
                cd = das_direction(
                    extractor,
                    [(p[0]["prompt"], p[1]["prompt"]) for p in pairs],
                    [(p[0]["subject"], p[1]["subject"]) for p in pairs],
                    layer=layer,
                    concept_name=concept_name,
                    module_type=module_type,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            bank.add(cd)
            print(f"  Extracted {method} direction for {concept_name} @ layer {layer}")

    return bank