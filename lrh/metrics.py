"""
Quantitative metrics for ROME x LRH analysis.

All functions are pure computations: tensors in, scalars/dicts out.

Metrics fall into four categories:
    1. Direction alignment (cosine similarity, projection magnitude)
    2. Subspace geometry (overlap via principal angles, Grassmann distance)
    3. Probe coherence (accuracy deltas pre/post edit)
    4. Predictive correlations (structural measures vs edit success)

Dimensional conventions:
    - Residual stream vectors: R^{d_model} (1600 for GPT-2 XL)
    - MLP key space vectors: R^{d_inner} (6400 for GPT-2 XL)
    - All vectors in a comparison must share the same dimensionality.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats as scipy_stats


def direction_alignment(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Cosine similarity between two direction vectors.

    Args:
        v1, v2: Vectors in the same space (both R^{d_model} or both R^{d_inner}).

    Returns:
        Cosine similarity in [-1, 1].
    """
    v1, v2 = v1.float().cpu(), v2.float().cpu()
    return (torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-10)).item()


def projection_magnitude(v: torch.Tensor, onto: torch.Tensor) -> float:
    """
    Magnitude of the projection of v onto the direction of `onto`.

    Returns |<v, onto/||onto||>| / ||v||, i.e., the fraction of v's norm
    along the `onto` direction. Value in [0, 1].
    """
    v, onto = v.float(), onto.float()
    onto_unit = onto / (onto.norm() + 1e-10)
    proj = torch.dot(v, onto_unit).abs()
    return (proj / (v.norm() + 1e-10)).item()


def subspace_overlap(
    basis_A: torch.Tensor,
    basis_B: torch.Tensor,
) -> float:
    """
    Subspace overlap via squared cosines of principal angles.

    Given two subspaces spanned by the rows of basis_A (k x d) and
    basis_B (m x d), computes the average squared cosine of the
    min(k, m) principal angles.

    Returns:
        Value in [0, 1]. 1 = identical subspaces, 0 = orthogonal.

    Reference:
        Golub & Van Loan (2013). Matrix Computations, §6.4.3.
    """
    # Orthonormalize both bases
    Q_A, _ = torch.linalg.qr(basis_A.T.float())  # (d, k)
    Q_B, _ = torch.linalg.qr(basis_B.T.float())  # (d, m)

    # Singular values of Q_A^T @ Q_B are cosines of principal angles
    cos_angles = torch.linalg.svdvals(Q_A.T @ Q_B)
    return (cos_angles**2).mean().item()


def grassmann_distance(
    basis_A: torch.Tensor,
    basis_B: torch.Tensor,
) -> float:
    """
    Grassmann distance between two subspaces.

    Defined as sqrt(sum of squared principal angles). Lower = more similar.

    Reference:
        Edelman et al. (1998). The geometry of algorithms with orthogonality
        constraints. SIAM J. Matrix Anal. Appl.
    """
    Q_A, _ = torch.linalg.qr(basis_A.T.float())
    Q_B, _ = torch.linalg.qr(basis_B.T.float())

    cos_angles = torch.linalg.svdvals(Q_A.T @ Q_B)
    cos_angles = cos_angles.clamp(-1.0, 1.0)
    angles = torch.acos(cos_angles)
    return angles.norm().item()


def explained_variance_by_directions(
    v: torch.Tensor,
    directions: torch.Tensor,
    orthogonalize: bool = True,
) -> Dict[str, float]:
    """
    Compute how much of ||v||^2 is explained by a set of directions.

    Args:
        v: Target vector, shape (d,).
        directions: Matrix of direction vectors, shape (k, d).
        orthogonalize: If True, apply Gram-Schmidt before projecting.

    Returns:
        {
            'total_explained': fraction of ||v||^2 captured (in [0, 1]),
            'per_direction': list of per-direction contributions,
            'residual_norm': ||v - projection||,
        }
    """
    v = v.float()
    dirs = directions.float()

    if orthogonalize and dirs.size(0) > 1:
        Q, _ = torch.linalg.qr(dirs.T)  # (d, k)
        dirs = Q.T  # (k, d), orthonormal rows
    else:
        dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-10)

    projections = dirs @ v  # (k,)
    per_dir = (projections**2 / (v.norm() ** 2 + 1e-10)).tolist()
    total = sum(per_dir)
    residual = v - dirs.T @ projections
    return {
        "total_explained": min(total, 1.0),
        "per_direction": per_dir,
        "residual_norm": residual.norm().item(),
    }


def probe_coherence_delta(
    pre_metrics: Dict[int, Dict[str, float]],
    post_metrics: Dict[int, Dict[str, float]],
    edit_layer: int = 17,
) -> Dict[str, object]:
    """
    Summarize probe coherence change across layers.

    Args:
        pre_metrics: layer -> {'accuracy': float, ...} before ROME edit.
        post_metrics: layer -> {'accuracy': float, ...} after ROME edit.
        edit_layer: The layer ROME targets.

    Returns:
        Summary dict with mean/max delta, affected layers, and per-layer breakdown.
    """
    deltas = {}
    for layer in pre_metrics:
        if layer in post_metrics:
            deltas[layer] = (
                post_metrics[layer]["accuracy"] - pre_metrics[layer]["accuracy"]
            )

    delta_values = list(deltas.values())
    affected = [l for l, d in deltas.items() if d < -0.05]

    return {
        "mean_delta_accuracy": float(np.mean(delta_values)) if delta_values else 0.0,
        "max_accuracy_drop": float(min(delta_values)) if delta_values else 0.0,
        "affected_layers": affected,
        "critical_layer_delta": deltas.get(edit_layer, None),
        "per_layer_delta": deltas,
    }


def d_prime(
    target_scores: torch.Tensor,
    nontarget_scores: torch.Tensor,
) -> float:
    """
    d' (d-prime) discriminability between two distributions.

    d' = (mu_target - mu_nontarget) / sqrt(0.5 * (sigma_target^2 + sigma_nontarget^2))

    Used to measure how selectively ROME's u vector activates for the
    target subject vs other entities.
    """
    mu_t = target_scores.float().mean()
    mu_n = nontarget_scores.float().mean()
    var_t = target_scores.float().var()
    var_n = nontarget_scores.float().var()
    pooled_std = (0.5 * (var_t + var_n)).sqrt()
    return ((mu_t - mu_n) / (pooled_std + 1e-10)).item()


def edit_success_correlation(
    structural_measures: Dict[str, List[float]],
    success_measures: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute Spearman correlations between LRH structural measures
    and ROME edit success metrics.

    Args:
        structural_measures: {measure_name: [value_per_edit, ...]}.
            E.g., 'v_concept_coverage', 'gate_selectivity', ...
        success_measures: {metric_name: [value_per_edit, ...]}.
            E.g., 'efficacy', 'paraphrase_success', ...

    Returns:
        Nested dict: structural_name -> metric_name -> {'rho', 'p_value'}.
    """
    results = {}
    for s_name, s_vals in structural_measures.items():
        results[s_name] = {}
        for m_name, m_vals in success_measures.items():
            rho, p = scipy_stats.spearmanr(s_vals, m_vals)
            results[s_name][m_name] = {"rho": float(rho), "p_value": float(p)}
    return results