"""Training entry excerpt for RP-STDS-DTR.

This file shows the experiment protocol and reported metrics. Dataset builders,
model registration, preprocessing, exact split manifests, checkpoint utilities,
and structural pruning operators remain in the internal project package.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch import Tensor

# Public excerpt modules in this folder.
from loss import PhysicsGuidedObjective
from model import RPSTDSTR
from temporal_transformer import SCALES as TEMPORAL_SCALES

# Internal research-project components are intentionally not distributed.
from rp_stds_dtr.data import build_complete_sequence_loaders
from rp_stds_dtr.pruning import (
    apply_structural_scale_pruning,
    rank_temporal_scales,
)
from rp_stds_dtr.utils.checkpoint import save_experiment_checkpoint
from rp_stds_dtr.utils.config import load_experiment_config


CLASS_NAMES = ("target", "decoy", "debris")


@dataclass(frozen=True)
class TrainSettings:
    sequence_length: int = 100
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    gamma: float = 0.15
    delta: float = 0.45
    lambda_time: float = 0.4
    lambda_frequency: float = 0.4
    lambda_entropy: float = 0.2
    pruning_tolerance: float = 0.005
    maximum_pruned_scales: int = 3
    seed: int = 42


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch(batch, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    radiation, image, label = batch
    return (
        radiation.to(device, non_blocking=True),
        image.to(device, non_blocking=True),
        label.to(device, non_blocking=True),
    )


def one_vs_rest_rates(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-class specificity and false-alarm rate."""
    total = matrix.sum()
    specificity, false_alarm_rate = [], []
    for class_index in range(len(CLASS_NAMES)):
        true_positive = matrix[class_index, class_index]
        false_negative = matrix[class_index, :].sum() - true_positive
        false_positive = matrix[:, class_index].sum() - true_positive
        true_negative = total - true_positive - false_negative - false_positive
        denominator = max(true_negative + false_positive, 1)
        specificity.append(true_negative / denominator)
        false_alarm_rate.append(false_positive / denominator)
    return np.asarray(specificity), np.asarray(false_alarm_rate)


def compute_manuscript_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Metric definitions used by the manuscript tables.

    F1 is support-weighted. Precision, recall, specificity, FAR and AUC are
    three-class one-versus-rest macro averages.
    """
    matrix = confusion_matrix(labels, predictions, labels=range(3))
    specificity, false_alarm_rate = one_vs_rest_rates(matrix)
    binary_labels = label_binarize(labels, classes=range(3))

    try:
        macro_auc = roc_auc_score(
            binary_labels, probabilities, average="macro", multi_class="ovr"
        )
    except ValueError:
        macro_auc = float("nan")

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "macro_specificity": float(specificity.mean()),
        "macro_far": float(false_alarm_rate.mean()),
        "macro_auc": float(macro_auc),
    }


def train_one_epoch(
    model,
    loader: Iterable,
    objective,
    optimizer,
    device: torch.device,
    settings: TrainSettings,
) -> float:
    model.train()
    accumulated_loss, sample_count = 0.0, 0

    for batch in loader:
        radiation, image, label = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, scale_weights, learned_response = model(radiation, image)
        loss = objective(logits, label, learned_response, scale_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
        optimizer.step()

        accumulated_loss += float(loss.detach()) * label.size(0)
        sample_count += label.size(0)

    return accumulated_loss / max(sample_count, 1)


@torch.no_grad()
def evaluate(
    model,
    loader: Iterable,
    objective,
    device: torch.device,
    retained_scales: Optional[Sequence[int]] = None,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    model.eval()
    losses, labels_all, predictions_all, probabilities_all = [], [], [], []

    for batch in loader:
        radiation, image, label = move_batch(batch, device)
        logits, scale_weights, learned_response = model(
            radiation, image, retained_scales=retained_scales
        )
        loss = objective(logits, label, learned_response, scale_weights)
        probability = logits.softmax(dim=1)

        losses.append(float(loss) * label.size(0))
        labels_all.append(label.cpu())
        predictions_all.append(probability.argmax(dim=1).cpu())
        probabilities_all.append(probability.cpu())

    labels = torch.cat(labels_all).numpy()
    predictions = torch.cat(predictions_all).numpy()
    probabilities = torch.cat(probabilities_all).numpy()
    metrics = compute_manuscript_metrics(labels, predictions, probabilities)
    mean_loss = sum(losses) / max(len(labels), 1)
    matrix = confusion_matrix(labels, predictions, labels=range(3))
    return mean_loss, metrics, matrix


def select_scale_subset(
    model,
    validation_loader,
    objective,
    device: torch.device,
    settings: TrainSettings,
) -> Tuple[int, ...]:
    """Validation-only scale removal followed by weight renormalization."""
    _, baseline, _ = evaluate(model, validation_loader, objective, device)
    ranking = rank_temporal_scales(model.temporal.scale_logits)
    selected = tuple(range(len(TEMPORAL_SCALES)))

    for remove_count in range(1, settings.maximum_pruned_scales + 1):
        removed = set(ranking[:remove_count])
        retained = tuple(
            index for index in range(len(TEMPORAL_SCALES))
            if index not in removed
        )
        _, report, _ = evaluate(
            model, validation_loader, objective, device,
            retained_scales=retained,
        )
        if report["weighted_f1"] >= (
            baseline["weighted_f1"] - settings.pruning_tolerance
        ):
            selected = retained

    # The internal operator removes unused branches and reapplies Softmax to
    # the retained class-specific logits so that their weights sum to one.
    apply_structural_scale_pruning(model, selected, renormalize=True)
    return selected


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    settings = TrainSettings()
    set_reproducible_seed(settings.seed)
    device = torch.device(args.device)
    project_config = load_experiment_config(args.config)

    # The private builder filters incomplete target appearances and performs
    # group-wise 70/15/15 splitting before any temporal windows are created.
    train_loader, validation_loader, test_loader = \
        build_complete_sequence_loaders(
            root=args.data_root,
            config=project_config,
            sequence_length=settings.sequence_length,
            batch_size=settings.batch_size,
            split=(settings.train_ratio,
                   settings.validation_ratio,
                   settings.test_ratio),
            seed=settings.seed,
        )

    model = RPSTDSTR(classes=len(CLASS_NAMES)).to(device)
    objective = PhysicsGuidedObjective(
        gamma=settings.gamma,
        delta=settings.delta,
        lambda_time=settings.lambda_time,
        lambda_frequency=settings.lambda_frequency,
        lambda_entropy=settings.lambda_entropy,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    best_epoch, best_weighted_f1, best_state = -1, -1.0, None
    history = []
    for epoch in range(1, settings.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, objective, optimizer, device, settings
        )
        validation_loss, validation_metrics, _ = evaluate(
            model, validation_loader, objective, device
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **validation_metrics,
        })

        if validation_metrics["weighted_f1"] > best_weighted_f1:
            best_epoch = epoch
            best_weighted_f1 = validation_metrics["weighted_f1"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    retained_indices = select_scale_subset(
        model, validation_loader, objective, device, settings
    )
    test_loss, test_metrics, test_matrix = evaluate(
        model, test_loader, objective, device
    )

    result = {
        "settings": asdict(settings),
        "best_epoch": best_epoch,
        "retained_scales": [TEMPORAL_SCALES[i] for i in retained_indices],
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "confusion_matrix": test_matrix.tolist(),
        "history": history,
    }
    save_experiment_checkpoint(model, result, Path(args.output_dir))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RP-STDS-DTR")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/main"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summary = run_experiment(arguments)
    print(json.dumps(summary["test_metrics"], indent=2, ensure_ascii=False))


# Not included in this excerpt:
# - exact simulation/data-generation pipeline and calibrated sensor parameters;
# - dataset indexing, normalization, augmentation and complete split manifests;
# - internal model registry and low-level attention/residual implementations;
# - learning-rate scheduling, distributed training and experiment tracking;
# - structural graph rewriting used to export the pruned deployment model.
