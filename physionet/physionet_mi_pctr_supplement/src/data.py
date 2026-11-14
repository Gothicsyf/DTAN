from pathlib import Path
import numpy as np
PROJECT_ROOT=Path(__file__).resolve().parents[1]
SUBJECTS=tuple(range(1,11))
EXPECTED_SHAPE=(676, 64, 640)
EXPECTED_COUNTS=(230, 220, 226)
def ea_transform(x, reference=None):
    """Same relative-regularized Self-EA used by the current experiment.

    A provided reference is an already fitted whitening matrix. No covariance
    is fitted in that branch. This permits target-adaptation-only fitting.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or len(x) < 1 or x.shape[-1] < 2 or not np.isfinite(x).all():
        raise ValueError("EA expects finite [N,C,T] with N>=1 and T>=2")
    channels, points = x.shape[1:]
    if reference is None:
        covariance = np.zeros((channels, channels), dtype=np.float64)
        for trial in x:
            centered = trial - trial.mean(axis=1, keepdims=True)
            covariance += centered @ centered.T / max(1, points - 1)
        covariance /= len(x)
        scale = float(np.trace(covariance) / channels)
        if not np.isfinite(scale) or scale <= np.finfo(np.float64).tiny:
            raise ValueError("Cannot fit EA to zero-energy or degenerate covariance")
        covariance += np.eye(channels) * (1e-6 * scale)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-8 * scale)
        whitening = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    else:
        whitening = np.asarray(reference, dtype=np.float64)
        if whitening.shape != (channels, channels) or not np.isfinite(whitening).all():
            raise ValueError("EA reference must be a finite whitening matrix matching the channel count")
    aligned = np.empty(x.shape, dtype=np.float32)
    for index, trial in enumerate(x):
        aligned[index] = whitening @ trial
    if not np.isfinite(aligned).all():
        raise ValueError("EA produced nonfinite values")
    return aligned, whitening


def load_all(cache_dir=PROJECT_ROOT / "data" / "cache"):
    cache_dir = Path(cache_dir)
    arrays = tuple(np.load(cache_dir / name, mmap_mode="r", allow_pickle=False)
                   for name in ("pre_ea.npy", "labels.npy", "subject_ids.npy", "trial_ids.npy"))
    x, y, subjects, trials = arrays
    if x.shape != EXPECTED_SHAPE or x.dtype != np.float64:
        raise ValueError(f"Unexpected pre-EA cache shape/dtype: {x.shape}/{x.dtype}")
    if any(a.shape != (EXPECTED_SHAPE[0],) or a.dtype != np.int64 for a in (y, subjects, trials)):
        raise ValueError("Unexpected label/subject/trial array shape or dtype")
    if not np.array_equal(np.unique(subjects), SUBJECTS) or not np.array_equal(np.bincount(y, minlength=3), EXPECTED_COUNTS):
        raise ValueError("Unexpected selected subjects or class counts")
    return arrays


def load_aligned(cache_dir=PROJECT_ROOT / "data" / "cache"):
    x = np.load(Path(cache_dir) / "source_ea.npy", mmap_mode="r", allow_pickle=False)
    if x.shape != EXPECTED_SHAPE or x.dtype != np.float32:
        raise ValueError("Unexpected source-EA cache shape or dtype")
    return x


