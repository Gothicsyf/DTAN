"""Pre-EA waveform PCTR targets with maximum-distance normalization.

Targets are descriptive signal statistics, never learned-feature statistics.
The corrected user definition is: average trials within each class, flatten
those prototypes, take Euclidean distances, and divide by the maximum over
unique off-diagonal class pairs. A zero maximum produces an all-zero matrix.
Learned-feature distances use their own detached maximum; the supplied target
is not normalized again in the loss.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F


def signal_topology(
    x_pre_ea: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    num_classes: int = 3,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Return a detached float32 target plus auditable sufficient metadata.

    ``x_pre_ea`` is the unaugmented pre-EA signal corresponding to exactly
    these labels. Neither target-subject filtering nor sampling is performed
    here: the caller must supply the declared source-only indices. Missing
    classes return ``(None, metadata)``; no historical/EMA values fill them in.
    All signal statistics are computed in float64, including for float32
    inputs. A tensor input is detached before any statistic is computed.
    """
    if num_classes < 2:
        raise ValueError("At least two classes are required for pair geometry.")
    device = x_pre_ea.device if isinstance(x_pre_ea, torch.Tensor) else torch.device("cpu")
    if isinstance(x_pre_ea, torch.Tensor):
        x = x_pre_ea.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        x = np.asarray(x_pre_ea, dtype=np.float64)
    if isinstance(y, torch.Tensor):
        labels_raw = y.detach().cpu().numpy().reshape(-1)
    else:
        labels_raw = np.asarray(y).reshape(-1)
    if x.ndim < 2 or x.shape[0] != len(labels_raw):
        raise ValueError("Expected x[N,...] and exactly N labels.")
    if not np.isfinite(x).all() or not np.isfinite(labels_raw).all():
        raise ValueError("Non-finite signal or label in PCTR statistics.")
    labels = labels_raw.astype(np.int64)
    if not np.array_equal(labels_raw, labels):
        raise ValueError("PCTR labels must be integers, not probabilities.")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"PCTR labels must be in [0, {num_classes - 1}].")
    counts = np.bincount(labels, minlength=num_classes)
    metadata: dict[str, Any] = {
        "valid": bool(np.all(counts > 0)),
        "counts": counts.tolist(),
        "num_trials": len(labels),
        "input_shape": list(x.shape),
        "normalizer": None,
        "normalizer_type": "unique_off_diagonal_max",
        "statistics_dtype": "float64",
        "formula": "euclidean(class_mean_waveforms) / unique_pair_max; zero if max=0",
        "missing_classes": np.flatnonzero(counts == 0).tolist(),
    }
    if not metadata["valid"]:
        metadata["reason"] = "missing_class_skip"
        return None, metadata
    prototypes = np.stack([x[labels == c].mean(axis=0) for c in range(num_classes)])
    flattened = prototypes.reshape(num_classes, -1)
    distances = cdist(flattened, flattened, metric="euclidean")
    pair = np.triu_indices(num_classes, 1)
    normalizer = float(distances[pair].max())
    normalized = distances / normalizer if normalizer > 0 else np.zeros_like(distances)
    metadata.update(
        normalizer=normalizer,
        raw_distances=distances.tolist(),
        normalized_distances=normalized.tolist(),
        reason="ok" if normalizer > 0 else "all_class_means_identical",
    )
    target = torch.from_numpy(normalized.astype(np.float32)).to(device=device).detach()
    return target, metadata


def prototype_tat_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    topology: torch.Tensor | np.ndarray | None,
    num_classes: int = 3,
) -> torch.Tensor:
    """Smooth-L1 between max-normalized feature distances and a fixed target.

    Feature prototypes and Smooth-L1 are retained from the archived objective.
    The feature-pair maximum is detached. The target is already max-normalized
    by ``signal_topology`` and must not be renormalized by its mean: that would
    cancel the corrected maximum convention. Explicit target detachment also
    protects against a caller accidentally supplying a differentiable tensor.
    """
    if features.ndim != 2:
        raise ValueError("Expected feature matrix [batch, feature_dim].")
    labels = labels.to(device=features.device, dtype=torch.long).reshape(-1)
    if len(labels) != len(features):
        raise ValueError("Feature and label batch lengths differ.")
    counts = torch.bincount(labels, minlength=num_classes)
    if topology is None or torch.any(counts == 0):
        return features.sum() * 0.0
    prototypes = torch.stack([features[labels == c].mean(dim=0) for c in range(num_classes)])
    distances = torch.cdist(prototypes, prototypes)
    pair_mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    pair_distances = distances[pair_mask]
    maximum = pair_distances.max().detach()
    denominator = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    pair_distances = pair_distances / denominator
    target_matrix = torch.as_tensor(topology, device=distances.device, dtype=distances.dtype).detach()
    if tuple(target_matrix.shape) != (num_classes, num_classes):
        raise ValueError("PCTR topology shape does not match num_classes.")
    target = target_matrix[pair_mask]
    return F.smooth_l1_loss(pair_distances, target)
