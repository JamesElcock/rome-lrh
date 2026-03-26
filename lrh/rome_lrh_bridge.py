"""
Core bridge connecting ROME's model editing to the Linear Representation
Hypothesis.

This module provides the central analysis: decomposing ROME's rank-1
update (u ⊗ v) in terms of concept directions, measuring alignment,
and testing whether linear structure predicts edit success.

Mathematical framework:

    ROME modifies W_c_proj at layer L:
        W' = W + u ⊗ v

    where:
        u ∈ R^{d_inner} (6400): normalized, C⁻¹-adjusted subject key
        v ∈ R^{d_model} (1600): value vector optimized for new fact

    LRH predicts that concepts are encoded as linear directions in R^{d_model}.
    If concept directions c_1, ..., c_k form a basis for the "concept subspace",
    then v = v_concept + v_residual, where:
        v_concept = Σ (v · c_i) c_i    (interpretable component)
        v_residual = v - v_concept       (unexplained component)

    The fraction ||v_concept||² / ||v||² measures how much of ROME's edit
    is captured by known linear concept structure.

Research questions addressed:
    RQ2: Does v align with concept directions for the edited relation?
    RQ3: How does the rank-1 update decompose in concept subspaces?
    RQ5: Does alignment predict edit success?
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.compute_u import compute_u, get_inv_cov
from rome.compute_v import compute_v
from rome.rome_hparams import ROMEHyperParams
from rome.rome_main import execute_rome, get_context_templates
from util import nethook

from .concept_directions import ConceptDirection, ConceptDirectionBank
from .config import LRHConfig
from .extraction import ActivationExtractor
from .metrics import (
    d_prime,
    direction_alignment,
    explained_variance_by_directions,
    projection_magnitude,
)


@dataclass
class ROMELRHAnalysis:
    """
    Container for a complete ROME × LRH analysis of a single edit.

    Stores ROME vectors, concept decomposition, gate analysis, and summary metrics.
    """

    # The edit
    request: Dict
    layer: int

    # ROME vectors
    u_vector: torch.Tensor  # (d_inner,) = (6400,)
    v_vector: torch.Tensor  # (d_model,) = (1600,)

    # Concept decomposition of v
    v_decomposition: Dict = field(default_factory=dict)

    # Gate (u) analysis
    gate_analysis: Dict = field(default_factory=dict)

    # Summary metrics
    metrics: Dict[str, float] = field(default_factory=dict)


def extract_rome_edit_vectors(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Execute ROME to extract the (u, v) vectors without permanently modifying
    the model.

    This wraps execute_rome, which internally computes deltas and restores
    original weights (invariant: model state unchanged).

    Returns:
        Dict mapping weight_name -> (u_vector, v_vector).
        For a single-layer edit (typical), this has one entry.
    """
    deltas = execute_rome(model, tok, request, hparams)
    return deltas  # {weight_name: (u, v)}


def decompose_v_in_concept_basis(
    v_vector: torch.Tensor,
    concept_directions: List[ConceptDirection],
    orthogonalize: bool = True,
) -> Dict:
    """
    Decompose ROME's v vector in a concept direction basis.

    Given concept directions c_1, ..., c_k ∈ R^{d_model}:
        1. Optionally orthogonalize via QR to get orthonormal basis {e_i}
        2. Compute projections: α_i = ⟨v, e_i⟩
        3. Compute residual: v_res = v - Σ α_i · e_i
        4. Explained variance: 1 - ||v_res||² / ||v||²

    This measures what fraction of ROME's edit message is interpretable
    as movement along known concept directions.

    A high explained variance supports the hypothesis that ROME's edits
    operate within the model's linear concept structure.

    Args:
        v_vector: ROME's right vector, shape (d_model,).
        concept_directions: List of ConceptDirection objects in R^{d_model}.
        orthogonalize: Whether to orthonormalize directions first.

    Returns:
        {
            'projections': {concept_name: projection_magnitude},
            'projection_fractions': {concept_name: |α_i|²/||v||²},
            'total_explained_variance': float,
            'residual_norm': float,
            'residual_direction': torch.Tensor or None,
        }
    """
    v = v_vector.float()
    if not concept_directions:
        return {
            "projections": {},
            "projection_fractions": {},
            "total_explained_variance": 0.0,
            "residual_norm": v.norm().item(),
            "residual_direction": v / (v.norm() + 1e-10),
        }

    # Build direction matrix
    dir_matrix = torch.stack([cd.direction.float() for cd in concept_directions])
    names = [cd.concept_name for cd in concept_directions]

    ev = explained_variance_by_directions(v, dir_matrix, orthogonalize=orthogonalize)

    projections = {
        name: abs(direction_alignment(v, cd.direction))
        for name, cd in zip(names, concept_directions)
    }
    projection_fractions = dict(zip(names, ev["per_direction"]))

    residual = v.norm().item() * (1 - ev["total_explained"]) ** 0.5
    residual_dir = None
    if ev["residual_norm"] > 1e-8:
        # Compute the actual residual direction
        if orthogonalize and len(concept_directions) > 1:
            Q, _ = torch.linalg.qr(dir_matrix.T)
            proj = Q @ Q.T @ v
        else:
            proj = sum(
                torch.dot(v, cd.direction.float()) * cd.direction.float()
                for cd in concept_directions
            )
        res = v - proj
        residual_dir = res / (res.norm() + 1e-10)

    return {
        "projections": projections,
        "projection_fractions": projection_fractions,
        "total_explained_variance": ev["total_explained"],
        "residual_norm": ev["residual_norm"],
        "residual_direction": residual_dir,
    }


def compute_edit_gate_alignment(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    u_vector: torch.Tensor,
    target_prompts: List[str],
    target_subjects: List[str],
    other_prompts: List[str],
    other_subjects: List[str],
    layer: int,
    config: LRHConfig,
) -> Dict:
    """
    Analyze ROME's u vector (the "gate") selectivity.

    The edit fires proportionally to ⟨k, u⟩ where k is the input to c_proj.
    A well-targeted edit should have high ⟨k, u⟩ for the target subject
    and low ⟨k, u⟩ for other entities.

    This tests ROME's implicit assumption: that the covariance-adjusted
    key direction u is selective enough to avoid collateral damage.

    The d' (d-prime) metric from signal detection theory quantifies this:
        d' = (μ_target - μ_other) / √(0.5(σ²_target + σ²_other))

    Higher d' means more selective gating.

    Args:
        target_prompts/subjects: Prompts/subjects for the target entity.
        other_prompts/subjects: Prompts/subjects for unrelated entities.

    Returns:
        {
            'target_activations': tensor of ⟨k, u⟩ for target prompts,
            'other_activations': tensor of ⟨k, u⟩ for other prompts,
            'd_prime': float,
            'selectivity_ratio': mean_target / mean_other,
            'mean_target': float,
            'mean_other': float,
        }
    """
    extractor = ActivationExtractor(model, tok, config)
    u = u_vector.float()

    # Extract MLP key representations (input to c_proj)
    target_keys = extractor.extract_mlp_key(
        target_prompts, target_subjects, [layer], config.token_strategy
    )[layer].float()

    other_keys = extractor.extract_mlp_key(
        other_prompts, other_subjects, [layer], config.token_strategy
    )[layer].float()

    # Compute gate activations: ⟨k, u⟩ (all on CPU for compatibility)
    target_keys = target_keys.cpu()
    other_keys = other_keys.cpu()
    u_cpu = u.cpu()
    target_acts = target_keys @ u_cpu
    other_acts = other_keys @ u_cpu

    dp = d_prime(target_acts, other_acts)
    mean_t = target_acts.mean().item()
    mean_o = other_acts.mean().item()

    return {
        "target_activations": target_acts.cpu(),
        "other_activations": other_acts.cpu(),
        "d_prime": dp,
        "selectivity_ratio": mean_t / (abs(mean_o) + 1e-10),
        "mean_target": mean_t,
        "mean_other": mean_o,
    }


def full_rome_lrh_analysis(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    rome_hparams: ROMEHyperParams,
    lrh_config: LRHConfig,
    concept_bank: Optional[ConceptDirectionBank] = None,
    other_prompts: Optional[List[str]] = None,
    other_subjects: Optional[List[str]] = None,
) -> ROMELRHAnalysis:
    """
    Run a complete ROME × LRH analysis for a single edit request.

    Steps:
        1. Execute ROME to get u, v vectors (model unchanged afterward).
        2. Decompose v in the concept direction basis.
        3. Analyze u's gating selectivity.
        4. Compile summary metrics.

    Args:
        request: ROME edit request dict with 'prompt', 'subject', 'target_new'.
        rome_hparams: ROME hyperparameters.
        lrh_config: LRH analysis configuration.
        concept_bank: Pre-computed concept directions. If None, decomposition skipped.
        other_prompts/subjects: Prompts for non-target entities (for gate analysis).

    Returns:
        Complete ROMELRHAnalysis object.
    """
    layer = rome_hparams.layers[0]  # Typically 17

    # Step 1: Extract ROME vectors
    print(f"Extracting ROME edit vectors for: {request['subject']} → {request['target_new']['str']}")
    deltas = extract_rome_edit_vectors(model, tok, request, rome_hparams)

    # Get u, v from the first (usually only) weight
    weight_name = list(deltas.keys())[0]
    u_vector, v_vector = deltas[weight_name]

    # Step 2: Concept decomposition
    v_decomp = {}
    if concept_bank is not None:
        concept_dirs = list(concept_bank.get_layer(layer).values())
        if concept_dirs:
            # Filter to directions in the same space as v (d_model)
            concept_dirs = [cd for cd in concept_dirs if cd.dim == v_vector.size(0)]
            v_decomp = decompose_v_in_concept_basis(v_vector, concept_dirs)

    # Step 3: Gate analysis
    gate = {}
    if other_prompts is not None and other_subjects is not None:
        target_prompts = [request["prompt"]]
        target_subjects = [request["subject"]]
        gate = compute_edit_gate_alignment(
            model, tok, u_vector,
            target_prompts, target_subjects,
            other_prompts, other_subjects,
            layer, lrh_config,
        )

    # Step 4: Summary metrics
    metrics = {
        "u_norm": u_vector.norm().item(),
        "v_norm": v_vector.norm().item(),
        "layer": layer,
    }
    if v_decomp:
        metrics["v_explained_variance"] = v_decomp.get("total_explained_variance", 0.0)
        metrics["v_residual_norm"] = v_decomp.get("residual_norm", 0.0)
    if gate:
        metrics["gate_d_prime"] = gate.get("d_prime", 0.0)
        metrics["gate_selectivity"] = gate.get("selectivity_ratio", 0.0)

    return ROMELRHAnalysis(
        request=request,
        layer=layer,
        u_vector=u_vector,
        v_vector=v_vector,
        v_decomposition=v_decomp,
        gate_analysis=gate,
        metrics=metrics,
    )