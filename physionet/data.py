"""PhysioNet three-class MI, native 160 Hz, explicit EDF calibration and EA.

Raw EDFs are read-only. Preprocessing is performed per continuous run, then
trials are cut at the annotation onset. No resampling or invented baseline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path(r"D:\AAA科研文件\数据集\脑电数据集\PhysioNet EEGMMIDB")
SUBJECTS = tuple(s for s in range(1, 110) if s not in (88, 92, 100))
MI_RUNS = (4, 6, 8, 10, 12, 14)
LEFT_RIGHT_RUNS = (4, 8, 12)
EXPECTED_SHAPE = (7149, 64, 640)
EXPECTED_COUNTS = (2405, 2362, 2382)
CLASS_NAMES = ("left_fist", "right_fist", "both_feet")
PREPROCESSING = {
    "schema_version": 1,
    "dataset": "PhysioNet EEGMMIDB 1.0.0",
    "subject_ids": list(SUBJECTS),
    "excluded_subjects": [88, 92, 100],
    "subject_exclusion_reason": "Task EDF sampling frequency is 128 Hz, not native 160 Hz",
    "mi_runs": list(MI_RUNS),
    "raw_fs_hz": 160,
    "output_fs_hz": 160,
    "channels": 64,
    "epoch_seconds": [0.0, 4.0],
    "epoch_endpoint": "left-closed right-open",
    "epoch_points": 640,
    "bandpass": {"type": "Butterworth", "order": 4, "low_hz": 0.1,
                 "high_hz": 40.0, "phase": "zero", "implementation": "sosfiltfilt",
                 "scope": "each complete continuous EDF run before epoching"},
    "notch": None,
    "resampling": None,
    "physical_units": "microvolts, calibrated using EDF physical/digital min and max",
    "operation_order": ["EDF calibration to microvolts", "continuous-run bandpass", "annotation epoching"],
    "zscore": False,
    "baseline_subtraction": False,
    "label_mapping": {"R04/R08/R12:T1": 0, "R04/R08/R12:T2": 1, "R06/R10/R14:T2": 2},
    "class_names": list(CLASS_NAMES),
    "excluded_events": "T0 rest; T1 both fists in R06/R10/R14; events shorter than 4 s or beyond recording boundary",
    "cache_stage": "after bandpass, before all Euclidean alignment",
    "cache_dtype": "float64",
    "trial_ids": "zero-based selected-three-class candidate index within each subject; short-window exclusion leaves its ID unused",
    "source_ea": "Per-subject covariance over all retained trials; float32 convenience cache; never use held-out subject's cache as target input",
    "target_ea": "Runner fits only target adaptation signals and applies the same whitening to target test signals",
}




def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_edf_header(stream):
    header = stream.read(256)
    if len(header) != 256 or header[:8].strip() != b"0":
        raise ValueError("Expected an EDF version-0 header")
    ns = int(header[252:256])
    nrecords = int(header[236:244])
    record_seconds = float(header[244:252])
    header_bytes = int(header[184:192])
    if nrecords <= 0 or record_seconds <= 0 or header_bytes != 256 + ns * 256:
        raise ValueError("Unsupported EDF record or header dimensions")
    signal_header = stream.read(ns * 256)
    if len(signal_header) != ns * 256:
        raise ValueError("Truncated EDF signal headers")
    fields, offset = {}, 0
    for name, width in [("labels", 16), ("transducer", 80), ("units", 8),
                        ("physical_min", 8), ("physical_max", 8),
                        ("digital_min", 8), ("digital_max", 8),
                        ("prefilter", 80), ("samples_per_record", 8), ("reserved", 32)]:
        fields[name] = [signal_header[offset + i * width:offset + (i + 1) * width].decode("ascii").strip()
                        for i in range(ns)]
        offset += ns * width
    samples = list(map(int, fields["samples_per_record"]))
    ann = [i for i, label in enumerate(fields["labels"]) if label == "EDF Annotations"]
    eeg = [i for i in range(ns) if i not in ann]
    if len(ann) != 1 or len(eeg) != 64 or any(n <= 0 for n in samples):
        raise ValueError("Expected 64 EEG channels and one EDF annotation channel")
    return dict(ns=ns, nrecords=nrecords, record_seconds=record_seconds,
                header_bytes=header_bytes, samples=samples, fields=fields,
                annotation_indices=ann, eeg_indices=eeg,
                recording_seconds=nrecords * record_seconds,
                record_bytes=2 * sum(samples))


def _parse_annotation_blocks(blocks):
    events = []
    for block in blocks:
        for tal in block.split(b"\x00"):
            if not tal or b"\x14" not in tal:
                continue
            timing, *descriptions = tal.split(b"\x14")
            parts = timing.split(b"\x15")
            onset = float(parts[0])
            duration = float(parts[1]) if len(parts) > 1 else None
            for description in descriptions:
                if description:
                    events.append(dict(onset=onset, duration=duration, event=description.decode("utf-8")))
    return sorted(events, key=lambda event: event["onset"])


def read_edf(path):
    """Read EDF digitally, calibrate EEG to uV, and return real annotations."""
    with Path(path).open("rb") as stream:
        header = read_edf_header(stream)
        payload = stream.read()
    expected = header["nrecords"] * header["record_bytes"]
    if len(payload) != expected:
        raise ValueError(f"EDF payload size {len(payload)} differs from expected {expected}: {path}")
    samples = header["samples"]
    offsets = np.r_[0, np.cumsum(samples)]
    records = np.frombuffer(payload, dtype="<i2").reshape(header["nrecords"], sum(samples))
    fields = header["fields"]
    fs = [samples[c] / header["record_seconds"] for c in header["eeg_indices"]]
    if set(fs) != {160.0}:
        raise ValueError(f"Expected native 160 Hz on every EEG channel, found {set(fs)}: {path}")
    eeg = np.empty((64, header["nrecords"] * samples[header["eeg_indices"][0]]), dtype=np.float64)
    calibration = []
    for out_channel, channel in enumerate(header["eeg_indices"]):
        digital = records[:, offsets[channel]:offsets[channel + 1]].reshape(-1)
        dmin, dmax = float(fields["digital_min"][channel]), float(fields["digital_max"][channel])
        pmin, pmax = float(fields["physical_min"][channel]), float(fields["physical_max"][channel])
        if dmax <= dmin or pmax <= pmin:
            raise ValueError("Invalid EDF calibration ranges")
        unit = fields["units"][channel]
        unit_scale = {"uV": 1.0, "mV": 1e3, "V": 1e6}.get(unit)
        if unit_scale is None:
            raise ValueError(f"Unsupported EDF physical unit {unit!r}")
        slope = (pmax - pmin) / (dmax - dmin) * unit_scale
        intercept = pmin * unit_scale - dmin * slope
        eeg[out_channel] = digital.astype(np.float64) * slope + intercept
        calibration.append(dict(channel=fields["labels"][channel], native_unit=unit,
                                slope_uv_per_count=slope, intercept_uv=intercept,
                                digital_min=dmin, digital_max=dmax,
                                physical_min=pmin, physical_max=pmax))
    annotation_blocks = []
    for record in range(header["nrecords"]):
        for channel in header["annotation_indices"]:
            begin = record * header["record_bytes"] + 2 * offsets[channel]
            annotation_blocks.append(payload[begin:begin + 2 * samples[channel]])
    events = _parse_annotation_blocks(annotation_blocks)
    if not events or not np.isfinite(eeg).all():
        raise ValueError("EDF has no events or nonfinite calibrated EEG")
    return eeg, events, header, calibration


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


def load_metadata(cache_dir=PROJECT_ROOT / "data" / "cache"):
    return json.loads((Path(cache_dir) / "metadata.json").read_text(encoding="utf-8"))


def prepare_cache(raw_dir=DEFAULT_RAW, cache_dir=PROJECT_ROOT / "data" / "cache"):
    raw_dir, cache_dir = Path(raw_dir).resolve(), Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    required = [cache_dir / name for name in ("pre_ea.npy", "labels.npy", "subject_ids.npy",
                                             "trial_ids.npy", "source_ea.npy", "source_whitening.npy")]
    existing = cache_dir / "metadata.json"
    if existing.exists():
        report = load_metadata(cache_dir)
        if report["preprocessing"] != PREPROCESSING:
            raise ValueError("Cache preprocessing differs; use another cache directory")
        if not all(path.is_file() for path in required):
            raise ValueError("Completed metadata references missing cache arrays")
        for item in report["raw_files"]:
            path = raw_dir / item["relative_path"]
            if path.stat().st_size != item["bytes"]:
                raise ValueError(f"Raw input fingerprint changed: {path}")
        load_all(cache_dir)
        load_aligned(cache_dir)
        return report
    if any(path.exists() for path in required):
        raise ValueError("Incomplete cache found; refusing to overwrite existing arrays")
    pre = np.lib.format.open_memmap(required[0], mode="w+", dtype=np.float64, shape=EXPECTED_SHAPE)
    aligned_cache = np.lib.format.open_memmap(required[4], mode="w+", dtype=np.float32, shape=EXPECTED_SHAPE)
    whitening_cache = np.empty((len(SUBJECTS), 64, 64), dtype=np.float64)
    y = np.empty(EXPECTED_SHAPE[0], dtype=np.int64)
    subject_ids = np.empty_like(y)
    trial_ids = np.empty_like(y)
    sos = butter(4, [.1, 40.0], btype="bandpass", fs=160, output="sos")
    raw_files, trials, excluded, per_subject, run_quality = [], [], [], [], []
    channel_order = None
    position = 0
    for subject_index, subject in enumerate(SUBJECTS):
        first_row, candidate_id = position, 0
        for run in MI_RUNS:
            path = raw_dir / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"
            eeg, events, header, calibration = read_edf(path)
            channel_labels = [header["fields"]["labels"][c] for c in header["eeg_indices"]]
            if channel_order is None:
                channel_order = channel_labels
            if channel_order != channel_labels:
                raise ValueError(f"EEG channel order differs: {path}")
            raw_files.append(dict(subject=subject, run=run, relative_path=str(path.relative_to(raw_dir)),
                                  bytes=path.stat().st_size,
                                  fs_hz=160, calibration=calibration))
            filtered = sosfiltfilt(sos, eeg, axis=-1)
            if not np.isfinite(filtered).all():
                raise ValueError(f"Nonfinite bandpass output: {path}")
            run_quality.append(dict(subject=subject, run=run, samples=eeg.shape[-1],
                                    raw_min_uv=float(eeg.min()), raw_max_uv=float(eeg.max()),
                                    raw_std_uv=float(eeg.std()),
                                    minimum_channel_std_uv=float(eeg.std(axis=-1).min()),
                                    flat_channels=int(np.count_nonzero(eeg.std(axis=-1) == 0)),
                                    filtered_std_uv=float(filtered.std())))
            for event_index, event in enumerate(events):
                event_code = event["event"]
                if event_code not in ("T0", "T1", "T2"):
                    raise ValueError(f"Unexpected event {event_code}: {path}")
                if event_code == "T0" or (run not in LEFT_RIGHT_RUNS and event_code == "T1"):
                    continue
                label = (0 if event_code == "T1" else 1) if run in LEFT_RIGHT_RUNS else 2
                onset_samples = event["onset"] * 160
                start = int(round(onset_samples))
                if abs(onset_samples - start) > 1e-6:
                    raise ValueError(f"Annotation onset does not lie on a 160 Hz sample: {path}: {event}")
                stop = start + 640
                record = dict(subject=subject, trial=candidate_id, run=run, event_index=event_index,
                              event=event_code, onset_seconds=event["onset"], duration_seconds=event["duration"],
                              sample_start=start, sample_stop_exclusive=stop, label=label, class_name=CLASS_NAMES[label],
                              relative_file=str(path.relative_to(raw_dir)))
                candidate_id += 1
                if event["duration"] is None or event["duration"] < 4 - 1e-8 or start < 0 or stop > eeg.shape[-1]:
                    excluded.append(dict(**record, reason="insufficient_4s_active_annotation_or_recording_boundary"))
                    continue
                if position >= len(y):
                    raise ValueError("More eligible trials than expected")
                pre[position] = filtered[:, start:stop]
                y[position], subject_ids[position], trial_ids[position] = label, subject, record["trial"]
                trials.append(dict(row=position, **record))
                position += 1
        rows = slice(first_row, position)
        aligned, whitening = ea_transform(pre[rows])
        aligned_cache[rows] = aligned
        whitening_cache[subject_index] = whitening
        per_subject.append(dict(subject=subject, cache_rows=[first_row, position],
                                trials=position-first_row, class_counts=np.bincount(y[rows], minlength=3).tolist(),
                                pre_ea_std_uv=float(np.std(pre[rows])), source_ea_std=float(np.std(aligned)),
                                source_ea_finite=True, whitening_index=subject_index))
        pre.flush()
        aligned_cache.flush()
        print(json.dumps(dict(stage="prepare", subject=subject, completed_subjects=subject_index+1,
                              total_subjects=len(SUBJECTS), completed_trials=position,
                              counts=per_subject[-1]["class_counts"])), flush=True)
        _write_json(cache_dir / "preparation_status.json", dict(state="preparing", subject=subject,
                    completed_subjects=subject_index+1, completed_trials=position, expected_trials=len(y)))
    if position != len(y) or not np.array_equal(np.bincount(y, minlength=3), EXPECTED_COUNTS):
        raise ValueError(f"Unexpected selected-trial total/counts: {position}/{np.bincount(y, minlength=3)}")
    if len(excluded) != 1 or (excluded[0]["subject"], excluded[0]["run"], excluded[0]["onset_seconds"]) != (104, 8, 102.5):
        raise ValueError(f"Unexpected short-trial exclusions: {excluded}")
    del pre, aligned_cache
    for path, array in zip(required[1:4], [y, subject_ids, trial_ids]):
        np.save(path, array, allow_pickle=False)
    np.save(required[5], whitening_cache, allow_pickle=False)
    report = dict(preprocessing=PREPROCESSING, raw_directory=str(raw_dir), raw_files=raw_files,
                  channel_order=channel_order, cache_shape=list(EXPECTED_SHAPE), total_trials=position,
                  selected_subject_count=len(SUBJECTS), class_counts=np.bincount(y, minlength=3).tolist(),
                  subjects=per_subject, trials=trials, excluded_trials=excluded, run_quality=run_quality,
                  numpy_version=np.__version__, scipy_version=scipy.__version__,
                  all_raw_and_preprocessed_values_finite=True,
                  waveform_quality_policy="No automatic artifact/amplitude rejection; finite and channel checks only",
                  cache_files=[dict(name=path.name, bytes=path.stat().st_size) for path in required])
    _write_json(existing, report)
    _write_json(cache_dir / "preparation_status.json", dict(state="complete", completed_subjects=len(SUBJECTS),
                completed_trials=position, class_counts=report["class_counts"]))
    load_all(cache_dir)
    load_aligned(cache_dir)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "cache")
    args = parser.parse_args()
    report = prepare_cache(args.raw_dir, args.cache_dir)
    print(json.dumps(dict(state="complete", cache=str(args.cache_dir), shape=report["cache_shape"],
                          subjects=report["selected_subject_count"], class_counts=report["class_counts"])), flush=True)


if __name__ == "__main__":
    main()
