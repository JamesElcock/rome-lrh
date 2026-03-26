"""
Linear Relational Embeddings (LRE) for comparison with ROME.

Implements the core ideas from:
    Hernandez et al. (2024). Linearity of Relation Decoding in Transformer
    Language Models. ICLR.

An LRE for a relation R is a linear map W_R: R^{d_model} -> R^{d_model}
such that W_R(h_s) ≈ h_o, where h_s is the subject representation at an
early layer and h_o is the predicted object representation at a late layer.

Connection to ROME:
    ROME's rank-1 update ΔW = u ⊗ v modifies the MLP's c_proj matrix.
    The effective change in the residual stream for input k is:
        Δoutput = (k · u) · v

    This is a rank-1 linear operator from MLP key space to residual stream.
    The LRE, in contrast, operates entirely within the residual stream
    (d_model → d_model). Comparing ROME's implicit relation to the LRE
    tests whether ROME's edits are consistent with the model's own linear
    relational structure.

    Key comparison: the LRE's top singular direction (from SVD of W_R)
    should align with ROME's v vector if both capture the same relational
    information.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook

from .config import LRHConfig
from .extraction import ActivationExtractor
from .metrics import direction_alignment, subspace_overlap


@dataclass
class LinearRelationalEmbedding:
    """
    A Linear Relational Embedding for a specific relation.

    The LRE maps subject representations at subject_layer to object
    representations at object_layer:
        h_o_pred = W @ h_s + b

    Attributes:
        W: (d_model, d_model) linear map.
        b: (d_model,) bias vector.
        relation_id: e.g. "P103" (mother tongue).
        subject_layer: layer from which subject reps are taken.
        object_layer: layer at which object reps are predicted.
        fit_r2: R^2 of the linear fit (explained variance).
    """

    W: torch.Tensor  # (d_model, d_model)
    b: torch.Tensor  # (d_model,)
    relation_id: str
    subject_layer: int
    object_layer: int
    fit_r2: float = 0.0

    def predict(self, h_s: torch.Tensor) -> torch.Tensor:
        """
        Predict object representation from subject representation.

        Args:
            h_s: Subject representation, shape (..., d_model).

        Returns:
            Predicted object representation, shape (..., d_model).
        """
        return h_s.float() @ self.W.float().T + self.b.float()

    def low_rank(self, rank: int) -> "LinearRelationalEmbedding":
        """
        Return a rank-k approximation of the LRE via SVD.

        The rank-1 approximation is most directly comparable to ROME's
        rank-1 update.
        """
        U, S, Vh = torch.linalg.svd(self.W.float(), full_matrices=False)
        W_k = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
        return LinearRelationalEmbedding(
            W=W_k,
            b=self.b,
            relation_id=self.relation_id,
            subject_layer=self.subject_layer,
            object_layer=self.object_layer,
            fit_r2=self.fit_r2,
        )

    def top_singular_directions(
        self, k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return top-k singular directions of W.

        Returns:
            (U[:, :k], S[:k], Vh[:k, :]) where:
            - Columns of U are output (object) space directions
            - Rows of Vh are input (subject) space directions
            - S are singular values
        """
        U, S, Vh = torch.linalg.svd(self.W.float(), full_matrices=False)
        return U[:, :k], S[:k], Vh[:k, :]


def extract_lre(
    extractor: ActivationExtractor,
    prompts: List[str],
    subjects: List[str],
    targets: List[str],
    subject_layer: int,
    object_layer: int,
    config: LRHConfig,
    relation_id: str = "",
) -> LinearRelationalEmbedding:
    """
    Extract an LRE by fitting a linear model from subject representations
    to object representations.

    Protocol:
        1. For each (subject, target) pair:
           - Extract h_s at subject_layer at the subject's last token
           - Extract h_o at object_layer at the last token before the target
        2. Fit W, b via ridge regression: H_o = H_s @ W^T + b

    Ridge regression is used because d_model (1600) may exceed the number
    of samples, making ordinary least squares ill-conditioned.

    Args:
        prompts: List of prompt templates with {} for subject.
        subjects: List of subject strings.
        targets: List of target/object strings (used for context but not
            for extraction position — we extract at the last prompt token).
        subject_layer: Layer for subject representations (early-mid, e.g. 8).
        object_layer: Layer for object representations (late, e.g. 36).
        config: LRH configuration with ridge_alpha.
        relation_id: Human-readable relation identifier.

    Returns:
        Fitted LinearRelationalEmbedding.
    """
    # Extract subject representations at subject_layer
    H_s = extractor.extract_residual_stream(
        prompts, subjects, [subject_layer], token_strategy="subject_last"
    )[subject_layer].float()

    # Extract object representations at object_layer (at last prompt token)
    H_o = extractor.extract_residual_stream(
        prompts, subjects, [object_layer], token_strategy="last"
    )[object_layer].float()

    # Ridge regression: H_o = H_s @ W^T + b
    # Center the data
    H_s_mean = H_s.mean(dim=0)
    H_o_mean = H_o.mean(dim=0)
    H_s_c = H_s - H_s_mean
    H_o_c = H_o - H_o_mean

    # Solve: W^T = (H_s^T H_s + alpha I)^{-1} H_s^T H_o
    d = H_s_c.size(1)
    alpha = config.lre_ridge_alpha
    A = H_s_c.T @ H_s_c + alpha * torch.eye(d, device=H_s_c.device, dtype=H_s_c.dtype)
    B = H_s_c.T @ H_o_c
    W_T = torch.linalg.solve(A, B)  # (d_model, d_model)
    W = W_T.T  # (d_model, d_model)

    # Bias
    b = H_o_mean - W @ H_s_mean

    # Compute R^2
    H_o_pred = H_s @ W.T + b
    ss_res = ((H_o - H_o_pred) ** 2).sum().item()
    ss_tot = ((H_o - H_o_mean) ** 2).sum().item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    return LinearRelationalEmbedding(
        W=W.cpu(),
        b=b.cpu(),
        relation_id=relation_id,
        subject_layer=subject_layer,
        object_layer=object_layer,
        fit_r2=r2,
    )


def compare_lre_to_rome(
    lre: LinearRelationalEmbedding,
    rome_u: torch.Tensor,
    rome_v: torch.Tensor,
    W_proj: torch.Tensor,
    W_fc: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compare the LRE to ROME's rank-1 update.

    ROME's edit at c_proj: for input k, Δoutput = (k · u) · v.
    The LRE represents the full relation as W_R: h_s → h_o.

    We compare:
        1. v vs LRE's top output singular direction: Do both point toward
           the same direction in the residual stream?
        2. ROME's effective relation (projected into residual stream) vs LRE.
        3. LRE's ability to predict ROME's target output.

    Args:
        lre: Fitted LinearRelationalEmbedding.
        rome_u: ROME's left vector, shape (d_inner,).
        rome_v: ROME's right vector, shape (d_model,).
        W_proj: c_proj weight matrix, shape depends on model convention.
        W_fc: Optional c_fc weight matrix for full residual→residual projection.

    Returns:
        Dict of comparison metrics.
    """
    rome_v = rome_v.float().cpu()
    rome_u = rome_u.float().cpu()

    # LRE top singular directions (already on CPU from extract_lre)
    U_lre, S_lre, Vh_lre = lre.top_singular_directions(k=5)

    # 1. Alignment of v with LRE's top output directions
    v_lre_top1_cos = direction_alignment(rome_v, U_lre[:, 0])

    # Projection of v onto LRE's top-k output subspace
    v_in_lre = U_lre.T @ rome_v  # (k,)
    v_lre_coverage = (v_in_lre**2).sum().item() / (rome_v.norm() ** 2 + 1e-10).item()

    # 2. Spectral analysis: singular value distribution
    lre_rank1_energy = (S_lre[0] ** 2).item() / ((S_lre**2).sum().item() + 1e-10)

    # 3. If c_fc weight is available, compute ROME's effective relation in
    #    residual stream coordinates
    #    ROME maps: h_residual --(c_fc)--> k --(k·u)*v--> residual
    #    Effective: h_residual --> (W_fc @ h . u) * v = (u^T @ W_fc) ⊗ v
    effective_cos = None
    if W_fc is not None:
        W_fc = W_fc.float()
        # u^T @ W_fc gives direction in residual stream whose projection
        # through c_fc produces the key component along u
        u_residual = W_fc.T @ rome_u  # (d_model,)
        u_residual = u_residual / (u_residual.norm() + 1e-10)

        # ROME's effective rank-1 operator in residual space: u_residual ⊗ v
        # Compare to LRE's rank-1 approximation
        lre_r1 = lre.low_rank(1)
        U1, S1, V1h = lre_r1.top_singular_directions(1)

        effective_cos = direction_alignment(rome_v, U1[:, 0])

    return {
        "v_lre_top1_cosine": v_lre_top1_cos,
        "v_lre_topk_coverage": v_lre_coverage,
        "lre_rank1_energy": lre_rank1_energy,
        "lre_fit_r2": lre.fit_r2,
        "effective_relation_cosine": effective_cos,
        "lre_top_singular_value": S_lre[0].item(),
    }


def rome_as_implicit_lre(
    model: AutoModelForCausalLM,
    rome_u: torch.Tensor,
    rome_v: torch.Tensor,
    layer: int,
    mlp_proj_tmp: str = "transformer.h.{}.mlp.c_proj",
    mlp_fc_tmp: str = "transformer.h.{}.mlp.c_fc",
) -> torch.Tensor:
    """
    Derive ROME's effective linear map in residual stream coordinates.

    ROME modifies c_proj at a specific layer. The full MLP pathway is:
        h_residual --(c_fc)--> h_hidden --(GELU)--> h_key --(c_proj)--> h_out

    The GELU nonlinearity breaks exact linearity, but near the operating
    point we can linearize: GELU'(x) ≈ 1 for large positive x.

    In the linear approximation, ROME's effective map is:
        Δh_out = v · u^T · W_fc · h_residual  (ignoring GELU)

    This gives us a (d_model, d_model) matrix: v ⊗ (W_fc^T @ u).

    Returns:
        Effective LRE matrix of shape (d_model, d_model).
    """
    # Get c_fc weight
    fc_name = mlp_fc_tmp.format(layer) + ".weight"
    W_fc = nethook.get_parameter(model, fc_name).detach().float().cpu()

    # GPT-2 Conv1D stores weight as (in_features, out_features) = (d_model, d_inner)
    # Forward: output = input @ weight, so W_fc maps (batch, d_model) -> (batch, d_inner)
    # W_fc has shape (d_model, d_inner) = (1600, 6400)

    rome_u = rome_u.float().cpu()
    rome_v = rome_v.float().cpu()

    # u^T @ W_fc^T gives the "subject direction" in residual space
    # W_fc shape: (d_model, d_inner), so W_fc^T @ u: (d_inner, d_model)^T @ (d_inner,)
    # = (d_model, d_inner) @ (d_inner,) ... wait:
    # W_fc is (d_model, d_inner) in GPT-2's Conv1D, forward is input @ W_fc
    # So to map u back to residual: W_fc @ u would be wrong.
    # u is in R^{d_inner}, W_fc maps R^{d_model} -> R^{d_inner}
    # The pseudoinverse maps R^{d_inner} -> R^{d_model}: W_fc^T @ (W_fc @ W_fc^T)^{-1}
    # But for the effective relation, we want:
    #   Δh_out(h_in) = v · (u^T · (W_fc @ h_in))  [ignoring GELU]
    #   = v · (u^T @ W_fc) · h_in
    #   = v ⊗ (W_fc^T @ u)  as a matrix
    # Note: u^T @ W_fc has shape (1, d_model), which is (W_fc^T @ u)^T

    u_residual = W_fc.T @ rome_u  # Wrong: W_fc is (d_model, d_inner), W_fc.T is (d_inner, d_model)
    # u is (d_inner,), so W_fc.T @ u is... (d_inner, d_model) @ (d_inner,) -> needs to be
    # Actually: if W_fc has shape (d_model, d_inner):
    #   forward: h_hidden = h_in @ W_fc  →  h_in (batch, d_model) @ W_fc (d_model, d_inner) = (batch, d_inner)
    #   u^T @ h_hidden = u^T @ (h_in @ W_fc) = (u^T @ W_fc^T) @ h_in^T
    # For the per-sample case: u · h_hidden = u · (W_fc^T @ h_in) = (W_fc @ u) · h_in
    # So: effective_direction_in_residual = W_fc @ u  → but W_fc is (d_model, d_inner), u is (d_inner,)
    # This doesn't work dimensionally. Let me reconsider.
    # W_fc: (d_model, d_inner) means row i is the weights for output feature i.
    # h_hidden = h_in @ W_fc: (1, d_model) @ (d_model, d_inner) = (1, d_inner)
    # u · h_hidden = sum_j u_j * h_hidden_j = sum_j u_j * sum_i h_in_i * W_fc_ij
    #             = sum_i h_in_i * sum_j W_fc_ij * u_j
    #             = sum_i h_in_i * (W_fc @ u)_i   where (W_fc @ u) is...
    # W_fc is (d_model, d_inner), u is (d_inner,)
    # But d_model < d_inner (1600 < 6400), so W_fc @ u would be (d_model,). But that's not right
    # because W_fc rows have d_inner elements.
    # Wait: W_fc shape is (d_model, d_inner) = (1600, 6400).
    # W_fc[i, j] connects input feature i to output feature j.
    # So W_fc @ u means (1600, 6400) @ (6400,) = (1600,). This IS correct.
    # u · h_hidden = u · (h_in @ W_fc) = h_in · (W_fc @ u) where both are d_model-dim.

    u_residual = W_fc @ rome_u  # (d_model,)
    u_residual = u_residual / (u_residual.norm() + 1e-10)

    # Effective LRE: v ⊗ u_residual
    effective_W = rome_v.unsqueeze(1) @ u_residual.unsqueeze(0)  # (d_model, d_model)

    return effective_W