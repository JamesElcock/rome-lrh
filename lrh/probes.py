"""
Linear probes for concept classification in activation space.

Implements the methodology from:
    - Alain & Bengio (2016). Understanding intermediate layers using linear
      classifier probes. arXiv:1610.01644.
    - Belinkov (2022). Probing Classifiers: Promises, Shortcomings, and
      Advances. Computational Linguistics.

Probes operate in the residual stream (R^{d_model} = R^{1600} for GPT-2 XL)
or MLP key space (R^{d_inner} = R^{6400}). A probe's weight vector provides
a "concept direction" that can be compared to ROME's v vector.

The key experiment: train probes on the pre-edit model, then evaluate on
post-edit activations to measure whether ROME preserves linear structure.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook

from .config import LRHConfig
from .extraction import ActivationExtractor


class LinearProbe(nn.Module):
    """
    Linear classifier probe: h -> softmax(W @ h + b).

    For binary classification, the classification direction is W[1] - W[0],
    which gives the direction in activation space that most discriminates
    the two classes. This direction is directly comparable to concept
    directions and to ROME's v vector (when both are in R^{d_model}).

    Args:
        input_dim: Dimensionality of the activation space.
            1600 for residual stream, 6400 for MLP key space.
        n_classes: Number of concept classes.
    """

    def __init__(self, input_dim: int, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)
        self.input_dim = input_dim
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def direction(self) -> torch.Tensor:
        """
        For binary probes, return the classification direction (W[1] - W[0]).

        This is the normal vector to the decision hyperplane, pointing
        toward class 1. Its orientation in activation space reveals which
        linear direction encodes the concept.

        Shape: (input_dim,)
        """
        assert self.n_classes == 2, "Direction only defined for binary probes."
        w = self.linear.weight.detach()  # (2, input_dim)
        d = w[1] - w[0]
        return d / (d.norm() + 1e-10)


def train_probe(
    probe: LinearProbe,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    config: LRHConfig,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
) -> Dict[str, List[float]]:
    """
    Train a linear probe with Adam + weight decay (L2 regularization).

    The regularization is important: unregularized probes on high-dimensional
    spaces (d=1600) can overfit and produce misleading directions.

    Reference:
        Hewitt & Liang (2019). Designing and Interpreting Probes with Control
        Tasks. EMNLP. (Discusses regularization for probes.)

    Returns:
        Training history: {'train_loss', 'train_acc', 'val_acc'}.
    """
    device = X_train.device if X_train.is_cuda else "cpu"
    probe = probe.to(device)
    X_train = X_train.float().to(device)
    y_train = y_train.long().to(device)

    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=config.probe_lr,
        weight_decay=config.probe_weight_decay,
    )

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    n = X_train.size(0)
    bs = config.probe_batch_size

    for epoch in range(config.probe_epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        epoch_loss, epoch_correct = 0.0, 0

        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            logits = probe(X_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += (logits.argmax(1) == y_train[idx]).sum().item()

        history["train_loss"].append(epoch_loss / n)
        history["train_acc"].append(epoch_correct / n)

        if X_val is not None:
            val_metrics = evaluate_probe(probe, X_val.to(device), y_val.to(device))
            history["val_acc"].append(val_metrics["accuracy"])

    return history


def evaluate_probe(
    probe: LinearProbe,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate probe accuracy.

    Returns:
        {'accuracy', 'balanced_accuracy', 'per_class_acc'}.
    """
    device = next(probe.parameters()).device
    probe.eval()
    X_test = X_test.float().to(device)
    y_test = y_test.long().to(device)

    with torch.no_grad():
        preds = probe(X_test).argmax(1)

    correct = (preds == y_test).float()
    accuracy = correct.mean().item()

    # Balanced accuracy: mean of per-class accuracies
    classes = y_test.unique()
    per_class = {}
    for c in classes:
        mask = y_test == c
        if mask.sum() > 0:
            per_class[c.item()] = correct[mask].mean().item()

    balanced = np.mean(list(per_class.values())) if per_class else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "per_class_acc": per_class,
    }


def train_probes_across_layers(
    extractor: ActivationExtractor,
    prompts: List[str],
    subjects: List[str],
    labels: torch.Tensor,
    layers: List[int],
    config: LRHConfig,
    module_type: str = "residual",
    n_classes: int = 2,
) -> Dict[int, Tuple[LinearProbe, Dict]]:
    """
    Train a separate linear probe at each specified layer.

    Args:
        module_type: "residual" (d_model), "mlp_key" (d_inner), or "mlp_out" (d_model).
        n_classes: Number of concept classes.

    Returns:
        Dict mapping layer_idx -> (trained_probe, eval_metrics).
    """
    # Extract activations
    if module_type == "residual":
        all_acts = extractor.extract_residual_stream(
            prompts, subjects, layers, config.token_strategy
        )
    elif module_type == "mlp_key":
        all_acts = extractor.extract_mlp_key(
            prompts, subjects, layers, config.token_strategy
        )
    elif module_type == "mlp_out":
        all_acts = extractor.extract_mlp_output(
            prompts, subjects, layers, config.token_strategy
        )
    else:
        raise ValueError(f"Unknown module_type: {module_type}")

    # Train/test split
    n = len(prompts)
    n_train = int(n * 0.7)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(config.seed))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    results = {}
    for layer in layers:
        acts = all_acts[layer]  # (n, d)
        input_dim = acts.size(1)

        X_train, y_train = acts[train_idx], labels[train_idx]
        X_test, y_test = acts[test_idx], labels[test_idx]

        probe = LinearProbe(input_dim, n_classes)
        train_probe(probe, X_train, y_train, config, X_test, y_test)
        metrics = evaluate_probe(probe, X_test, y_test)
        results[layer] = (probe, metrics)

        print(
            f"  Layer {layer:2d}: accuracy={metrics['accuracy']:.3f}  "
            f"balanced={metrics['balanced_accuracy']:.3f}  "
            f"(dim={input_dim})"
        )

    return results


def compute_probe_coherence(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    probes: Dict[int, LinearProbe],
    test_prompts: List[str],
    test_subjects: List[str],
    test_labels: torch.Tensor,
    rome_request: Dict,
    rome_hparams: ROMEHyperParams,
    config: LRHConfig,
    module_type: str = "residual",
) -> Dict[int, Dict[str, float]]:
    """
    Measure probe coherence before and after a ROME edit.

    Protocol:
        1. Extract activations from the unedited model → evaluate probes (pre).
        2. Apply ROME edit to the model.
        3. Extract activations from the edited model → evaluate probes (post).
        4. Restore original weights.

    This measures whether ROME preserves existing linear representations.

    Returns:
        Dict mapping layer -> {
            'pre_accuracy', 'post_accuracy', 'delta_accuracy'
        }
    """
    extractor = ActivationExtractor(model, tok, config)
    layers = sorted(probes.keys())

    # Pre-edit evaluation
    if module_type == "residual":
        pre_acts = extractor.extract_residual_stream(
            test_prompts, test_subjects, layers, config.token_strategy
        )
    elif module_type == "mlp_key":
        pre_acts = extractor.extract_mlp_key(
            test_prompts, test_subjects, layers, config.token_strategy
        )
    else:
        pre_acts = extractor.extract_mlp_output(
            test_prompts, test_subjects, layers, config.token_strategy
        )

    pre_metrics = {}
    for layer in layers:
        pre_metrics[layer] = evaluate_probe(
            probes[layer], pre_acts[layer], test_labels
        )

    # Apply ROME edit
    edited_model, weights_copy = apply_rome_to_model(
        model, tok, [rome_request], rome_hparams,
        copy=False, return_orig_weights=True,
    )

    # Post-edit evaluation
    if module_type == "residual":
        post_acts = extractor.extract_residual_stream(
            test_prompts, test_subjects, layers, config.token_strategy
        )
    elif module_type == "mlp_key":
        post_acts = extractor.extract_mlp_key(
            test_prompts, test_subjects, layers, config.token_strategy
        )
    else:
        post_acts = extractor.extract_mlp_output(
            test_prompts, test_subjects, layers, config.token_strategy
        )

    post_metrics = {}
    for layer in layers:
        post_metrics[layer] = evaluate_probe(
            probes[layer], post_acts[layer], test_labels
        )

    # Restore original weights
    with torch.no_grad():
        for k, v in weights_copy.items():
            nethook.get_parameter(model, k)[...] = v.to(
                next(model.parameters()).device
            )

    # Compute deltas
    results = {}
    for layer in layers:
        results[layer] = {
            "pre_accuracy": pre_metrics[layer]["accuracy"],
            "post_accuracy": post_metrics[layer]["accuracy"],
            "delta_accuracy": (
                post_metrics[layer]["accuracy"] - pre_metrics[layer]["accuracy"]
            ),
        }

    return results