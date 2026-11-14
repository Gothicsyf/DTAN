"""Per-fold, pre-EA waveform prior for the four-class experiments."""
import json
import os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from pctr import signal_topology

def fold_prior(processed, subjects, labels, destination):
    processed = Path(processed).resolve()
    hgd = "hgd" in processed.name.lower() or "hgd" in __file__.lower()
    default = processed if "noea" in processed.name.lower() else processed.with_name("HGD_noea" if hgd else "BCIC_2a_noea")
    pre = Path(os.environ.get("PCTR_PRE_EA_DIR", default)).resolve()
    if pre == processed and "noea" not in pre.name.lower():
        raise ValueError("Supply pre-EA waveforms, not the aligned training cache")
    sums = None
    counts = np.zeros(4, dtype=np.int64)
    offset = 0
    for subject in subjects:
        for session in ("train", "test"):
            file = pre / f"S{subject}{session}" / "data.mat"
            if not file.is_file():
                raise FileNotFoundError(f"Missing pre-EA cache: {file}. Set PCTR_PRE_EA_DIR to the unaligned cache.")
            data = loadmat(file)
            x = np.asarray(data["x_data"])
            y = np.asarray(data["y_data"]).reshape(-1)
            np.testing.assert_array_equal(y, labels[offset:offset+len(y)])
            if len(x) != len(y) or not np.isin(y, range(4)).all():
                raise ValueError(f"Invalid trial or class count in {file}")
            offset += len(y)
            if sums is None:
                sums = np.zeros((4, *x.shape[1:]), dtype=np.float64)
            for c in range(4):
                group = x[y == c]
                sums[c] += group.sum(axis=0, dtype=np.float64)
                counts[c] += len(group)
    if offset != len(labels) or min(counts) == 0:
        raise ValueError("Incomplete source population")
    means = sums / counts.reshape((4,) + (1,) * (sums.ndim-1))
    matrix, meta = signal_topology(means, np.arange(4), 4)
    meta.update(counts=counts.tolist(), num_trials=offset, input_shape=[offset,*sums.shape[1:]], source_subjects=list(map(int, subjects)))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"pctr_source_prior.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    return matrix
