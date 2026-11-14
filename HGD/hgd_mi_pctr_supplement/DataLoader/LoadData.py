import os
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from scipy.signal import butter, filtfilt, iirnotch, resample
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import ConcatDataset, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EEGDataset(Dataset):
    def __init__(self, data_tensor, label_tensor):
        self.data_tensor = data_tensor.float()
        self.label_tensor = label_tensor.long().reshape(-1)

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


my_dataset = EEGDataset


def _as_path(path):
    return Path(path).expanduser().resolve()


def _load_mat_arrays(mat_path):
    mat = sio.loadmat(str(mat_path))
    x_key = "x_data" if "x_data" in mat else "X"
    y_key = "y_data" if "y_data" in mat else ("y_labels" if "y_labels" in mat else "y")
    if x_key not in mat or y_key not in mat:
        raise KeyError(f"{mat_path} must contain x_data/X and y_data/y_labels/y.")
    x = np.asarray(mat[x_key], dtype=np.float32)
    y = np.asarray(mat[y_key]).reshape(-1).astype(np.int64)
    return x, y


def load_processed_split(subject, data_path, split):
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")
    mat_path = _as_path(data_path) / f"S{subject}{split}" / "data.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {mat_path}")
    x, y = _load_mat_arrays(mat_path)
    return EEGDataset(torch.from_numpy(x), torch.from_numpy(y))


def get_test_EEG_data(sub, data_path):
    return load_processed_split(sub, data_path, "test")


def get_EEG_Hold_Out_Scheme(sub, data_path, validation_size=0.2, data_seed=20140101):
    train_path = _as_path(data_path) / f"S{sub}train" / "data.mat"
    train_x, train_y = _load_mat_arrays(train_path)

    split_train_x, split_validation_x, split_train_y, split_validation_y = train_test_split(
        train_x,
        train_y,
        test_size=validation_size,
        random_state=data_seed,
        stratify=train_y,
    )

    for label in np.unique(train_y):
        print(f"class-{label}:{np.sum(train_y == label)}")

    train_dataset = EEGDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    split_train_dataset = EEGDataset(torch.from_numpy(split_train_x), torch.from_numpy(split_train_y))
    split_validation_dataset = EEGDataset(torch.from_numpy(split_validation_x), torch.from_numpy(split_validation_y))
    test_dataset = get_test_EEG_data(sub, data_path)
    return train_dataset, split_train_dataset, split_validation_dataset, test_dataset


def get_single_subject_all_data(sub, data_path):
    train_x, train_y = _load_mat_arrays(_as_path(data_path) / f"S{sub}train" / "data.mat")
    test_x, test_y = _load_mat_arrays(_as_path(data_path) / f"S{sub}test" / "data.mat")
    x_all = np.concatenate([train_x, test_x], axis=0)
    y_all = np.concatenate([train_y, test_y], axis=0)
    return EEGDataset(torch.from_numpy(x_all), torch.from_numpy(y_all))


def get_loso_datasets(target_sub, data_path, total_subjects=9):
    source_datasets = [
        get_single_subject_all_data(sub, data_path)
        for sub in range(1, total_subjects + 1)
        if sub != target_sub
    ]
    train_dataset = ConcatDataset(source_datasets)
    test_dataset = get_single_subject_all_data(target_sub, data_path)
    return train_dataset, test_dataset


def get_EEG_LOSO_Scheme(target_sub, data_path, total_subjects=9):
    return get_loso_datasets(target_sub, data_path, total_subjects)


def get_other_Subject_EEG_data(sub, data_path, all_session=True, num_of_subject=9):
    datasets = []
    for subject in range(1, num_of_subject + 1):
        if subject == sub:
            continue
        if all_session:
            datasets.append(get_single_subject_all_data(subject, data_path))
        else:
            datasets.append(load_processed_split(subject, data_path, "train"))
    return ConcatDataset(datasets)


def get_one_Subject_EEG_test_session(sub, data_path):
    return get_test_EEG_data(sub, data_path)


def get_EEG_K_Fold_Scheme(sub, data_path, k=10, validation_size=0.2, data_seed=20240101, all_session=False):
    train_x, train_y = _load_mat_arrays(_as_path(data_path) / f"S{sub}train" / "data.mat")
    if all_session:
        test_x, test_y = _load_mat_arrays(_as_path(data_path) / f"S{sub}test" / "data.mat")
        train_x = np.concatenate([train_x, test_x], axis=0)
        train_y = np.concatenate([train_y, test_y], axis=0)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=data_seed)
    folds = []
    for train_index, test_index in skf.split(train_x, train_y):
        fold_train_x = train_x[train_index]
        fold_train_y = train_y[train_index]
        fold_test_x = train_x[test_index]
        fold_test_y = train_y[test_index]
        split_train_x, split_validation_x, split_train_y, split_validation_y = train_test_split(
            fold_train_x,
            fold_train_y,
            test_size=validation_size,
            random_state=data_seed,
            stratify=fold_train_y,
        )
        folds.append(
            [
                EEGDataset(torch.from_numpy(fold_train_x), torch.from_numpy(fold_train_y)),
                EEGDataset(torch.from_numpy(split_train_x), torch.from_numpy(split_train_y)),
                EEGDataset(torch.from_numpy(split_validation_x), torch.from_numpy(split_validation_y)),
                EEGDataset(torch.from_numpy(fold_test_x), torch.from_numpy(fold_test_y)),
            ]
        )
    return folds


def euclidean_alignment(x_data, epsilon=1e-6):
    print("  -> Applying Self-EA...", end="")
    trials, channels, time_points = x_data.shape
    covariances = np.zeros((trials, channels, channels), dtype=np.float64)
    for i in range(trials):
        x = x_data[i] - np.mean(x_data[i], axis=1, keepdims=True)
        covariances[i] = np.dot(x, x.T) / max(1, time_points - 1)
    reference = covariances.mean(axis=0) + np.eye(channels) * epsilon
    try:
        eigvals, eigvecs = np.linalg.eigh(reference)
        eigvals = np.maximum(eigvals, 1e-8)
        whitening = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
    except np.linalg.LinAlgError:
        print("failed; using original data.")
        return x_data

    aligned = np.zeros_like(x_data)
    for i in range(trials):
        aligned[i] = whitening @ x_data[i]
    print("Done.")
    return aligned


def _require_mne():
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required only when building raw datasets.") from exc
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    return mne


class Load_IV2a:
    def __init__(self, raw_root=None, save_root=None, no_ea=False):
        self.file_to_load = Path(raw_root) if raw_root else PROJECT_ROOT / "RawData" / "BCICIV_2a"
        self.save_root = Path(save_root) if save_root else PROJECT_ROOT / "Data" / "BCIC_2a"
        self.channels_to_remove = ["EOG-left", "EOG-central", "EOG-right"]
        self.no_ea = no_ea
        self.fs = 250

    def _apply_classic_filter(self, data, fs):
        nyq = 0.5 * fs
        b, a = butter(4, [0.1 / nyq, 40.0 / nyq], btype="bandpass")
        data = filtfilt(b, a, data, axis=1)
        b_notch, a_notch = iirnotch(50.0, 30.0, fs)
        return filtfilt(b_notch, a_notch, data, axis=1)

    def get_epochs(self, tmin=-0.5, tmax=4.0, baseline=None, file_path=None):
        mne = _require_mne()
        raw_data = mne.io.read_raw_gdf(str(file_path), preload=True, verbose=False)
        self.fs = raw_data.info.get("sfreq")
        picks = mne.pick_types(raw_data.info, eeg=True, eog=False, stim=False)
        data_np = self._apply_classic_filter(raw_data.get_data(picks=picks), self.fs)
        raw_data._data[picks, :] = data_np
        events, event_ids = mne.events_from_annotations(raw_data, verbose=False)
        stims = [value for key, value in event_ids.items() if key in ["769", "770", "771", "772"]]
        epochs = mne.Epochs(
            raw_data,
            events,
            event_id=stims,
            tmin=tmin,
            tmax=tmax,
            event_repeated="drop",
            baseline=baseline,
            preload=True,
            proj=False,
            reject_by_annotation=True,
            verbose=False,
        )
        epochs = epochs.drop_channels(self.channels_to_remove, on_missing="ignore")
        return {"x_data": epochs.get_data() * 1e6, "y_labels": epochs.events[:, -1] - min(epochs.events[:, -1])}

    def get_session2(self, tmin=-0.5, tmax=4.0, baseline=None, file_path=None, label_path=None):
        mne = _require_mne()
        raw_data = mne.io.read_raw_gdf(str(file_path), preload=True, verbose=False)
        self.fs = raw_data.info.get("sfreq")
        picks = mne.pick_types(raw_data.info, eeg=True, eog=False, stim=False)
        data_np = self._apply_classic_filter(raw_data.get_data(picks=picks), self.fs)
        raw_data._data[picks, :] = data_np
        events, event_ids = mne.events_from_annotations(raw_data, verbose=False)
        stims = [value for key, value in event_ids.items() if key in ["783"]]
        epochs = mne.Epochs(
            raw_data,
            events,
            event_id=stims,
            tmin=tmin,
            tmax=tmax,
            event_repeated="drop",
            baseline=baseline,
            preload=True,
            proj=False,
            reject_by_annotation=False,
            verbose=False,
        )
        epochs = epochs.drop_channels(self.channels_to_remove, on_missing="ignore")
        labels = sio.loadmat(str(label_path))["classlabel"].reshape(-1) - 1
        return {"x_data": epochs.get_data() * 1e6, "y_labels": labels}

    def build_dataset(self):
        self.save_root.mkdir(parents=True, exist_ok=True)
        for sub in range(1, 10):
            print(f"\nProcessing IV-2a subject {sub}...")
            train = self.get_epochs(file_path=self.file_to_load / f"A0{sub}T.gdf")
            test = self.get_session2(
                file_path=self.file_to_load / f"A0{sub}E.gdf",
                label_path=self.file_to_load / f"A0{sub}E.mat",
            )
            train_x = np.asarray(train["x_data"])[:, :, :1125]
            test_x = np.asarray(test["x_data"])[:, :, :1125]
            if not self.no_ea:
                train_x = euclidean_alignment(train_x)
                test_x = euclidean_alignment(test_x)
            train_y = np.asarray(train["y_labels"]).reshape(-1)
            test_y = np.asarray(test["y_labels"]).reshape(-1)
            self._save_subject(sub, train_x, train_y, test_x, test_y)

    def _save_subject(self, sub, train_x, train_y, test_x, test_y):
        train_dir = self.save_root / f"S{sub}train"
        test_dir = self.save_root / f"S{sub}test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        sio.savemat(str(train_dir / "data.mat"), {"x_data": train_x, "y_data": train_y})
        sio.savemat(str(test_dir / "data.mat"), {"x_data": test_x, "y_data": test_y})
        print(f"  Saved train {train_x.shape}, test {test_x.shape}")


class Load_IV2b:
    def __init__(self, raw_root=None, save_root=None):
        self.file_to_load = Path(raw_root) if raw_root else PROJECT_ROOT / "RawData" / "BCICIV_2b"
        self.save_root = Path(save_root) if save_root else PROJECT_ROOT / "Data" / "BCIC_2b"
        self.fs = 250

    def _apply_classic_filter(self, data, fs):
        nyq = 0.5 * fs
        b, a = butter(4, [0.1 / nyq, 40.0 / nyq], btype="bandpass")
        data = filtfilt(b, a, data, axis=1)
        b_notch, a_notch = iirnotch(50.0, 30.0, fs)
        return filtfilt(b_notch, a_notch, data, axis=1)

    def get_single_file_data(self, gdf_path, label_path=None, is_train=True, tmin=-0.5, tmax=4.0):
        mne = _require_mne()
        raw_data = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
        self.fs = raw_data.info.get("sfreq")
        picks = mne.pick_types(raw_data.info, eeg=True, eog=False, stim=False, exclude="bads")
        if len(picks) > 3:
            picks = picks[:3]
        data_np = self._apply_classic_filter(raw_data.get_data(picks=picks), self.fs)
        raw_data._data[picks, :] = data_np
        events, event_ids = mne.events_from_annotations(raw_data, verbose=False)

        if is_train:
            stims = [value for key, value in event_ids.items() if key in ["769", "770"]]
            epochs = mne.Epochs(
                raw_data,
                events,
                event_id=stims,
                tmin=tmin,
                tmax=tmax,
                event_repeated="drop",
                baseline=None,
                preload=True,
                proj=False,
                reject_by_annotation=True,
                verbose=False,
                picks=picks,
            )
            current_ids = epochs.events[:, -1]
            val_769 = event_ids.get("769")
            labels = np.zeros(len(current_ids), dtype=np.int64)
            labels[current_ids != val_769] = 1
        else:
            stims = [value for key, value in event_ids.items() if key in ["783"]]
            if not stims:
                stims = [value for key, value in event_ids.items() if key in ["769", "770"]]
            epochs = mne.Epochs(
                raw_data,
                events,
                event_id=stims,
                tmin=tmin,
                tmax=tmax,
                event_repeated="drop",
                baseline=None,
                preload=True,
                proj=False,
                reject_by_annotation=False,
                verbose=False,
                picks=picks,
            )
            if label_path is None or not Path(label_path).exists():
                raise FileNotFoundError(f"Label file not found: {label_path}")
            labels = sio.loadmat(str(label_path))["classlabel"].reshape(-1)
            labels = labels - 1 if labels.min() == 1 else labels
            if len(labels) != len(epochs):
                min_len = min(len(labels), len(epochs))
                labels = labels[:min_len]
                epochs = epochs[:min_len]

        return epochs.get_data() * 1e6, labels

    def build_dataset(self):
        self.save_root.mkdir(parents=True, exist_ok=True)
        for sub in range(1, 10):
            print(f"\nProcessing IV-2b subject {sub}...")
            train_x_list, train_y_list = [], []
            for sess in [1, 2, 3]:
                gdf = self.file_to_load / f"B0{sub}0{sess}T.gdf"
                if gdf.exists():
                    x, y = self.get_single_file_data(gdf, is_train=True)
                    train_x_list.append(x)
                    train_y_list.append(y)
                else:
                    print(f"  Missing file: {gdf.name}")
            test_x_list, test_y_list = [], []
            for sess in [4, 5]:
                gdf = self.file_to_load / f"B0{sub}0{sess}E.gdf"
                mat = self.file_to_load / f"B0{sub}0{sess}E.mat"
                if gdf.exists():
                    x, y = self.get_single_file_data(gdf, label_path=mat, is_train=False)
                    test_x_list.append(x)
                    test_y_list.append(y)
                else:
                    print(f"  Missing file: {gdf.name}")
            if not train_x_list or not test_x_list:
                raise FileNotFoundError(f"Incomplete IV-2b raw files for subject {sub}.")
            train_x = euclidean_alignment(np.concatenate(train_x_list, axis=0)[:, :, :1125])
            test_x = euclidean_alignment(np.concatenate(test_x_list, axis=0)[:, :, :1125])
            train_y = np.concatenate(train_y_list, axis=0).reshape(-1)
            test_y = np.concatenate(test_y_list, axis=0).reshape(-1)
            Load_IV2a._save_subject(self, sub, train_x, train_y, test_x, test_y)


class Load_HGD:
    def __init__(self, raw_root=None, save_root=None, n_channels=44):
        self.root_path = Path(raw_root) if raw_root else PROJECT_ROOT / "RawData" / "HGD"
        self.save_root = Path(save_root) if save_root else PROJECT_ROOT / "Data" / "HGD_Processed"
        self.target_fs = 250
        self.high_pass = 0.5
        self.tmin = 0.0
        self.tmax = 4.5
        self.n_channels = int(n_channels)
        self.target_channels = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
            "T7", "C3", "Cz", "C4", "T8", "TP7", "CP5", "CP1", "CP2", "CP6", "TP8",
            "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2", "AF7", "AF3",
            "AF4", "AF8", "F5", "F1", "F2", "F6", "FT7", "FT8", "FCz", "CPz",
            "FC3", "FC4", "C5", "C1", "C2", "C6", "CP3", "CP4",
        ]

    def _pick_motor_channels(self, raw):
        mne = _require_mne()
        clean_map = {name: name.replace("EEG ", "").replace(".", "").strip() for name in raw.ch_names}
        picks = []
        for target in self.target_channels:
            for raw_name, clean in clean_map.items():
                if clean == target and raw_name not in picks:
                    picks.append(raw_name)
                    break
            if len(picks) >= self.n_channels:
                break
        if len(picks) < self.n_channels:
            priority = []
            fallback = []
            for raw_name, clean in clean_map.items():
                if raw_name in picks:
                    continue
                if clean.startswith(("FC", "C", "CP")):
                    priority.append(raw_name)
                else:
                    fallback.append(raw_name)
            picks = (picks + priority + fallback)[: self.n_channels]
        return mne.pick_channels(raw.info["ch_names"], include=picks)

    def _preprocess_raw(self, raw):
        raw.load_data()
        picks = self._pick_motor_channels(raw)
        raw.filter(l_freq=self.high_pass, h_freq=120.0, picks=picks, n_jobs=1, verbose=False)
        if raw.info["sfreq"] != self.target_fs:
            raw.resample(self.target_fs, npad="auto")
        return raw, picks

    def get_epochs_from_edf(self, file_path):
        mne = _require_mne()
        raw = mne.io.read_raw_edf(str(file_path), preload=False, verbose=False)
        raw, picks = self._preprocess_raw(raw)
        events, event_ids = mne.events_from_annotations(raw, verbose=False)
        keywords = ["1", "2", "3", "4", "left", "right", "hand", "foot", "feet", "rest", "class"]
        target_values = [value for key, value in event_ids.items() if any(k in key.lower() for k in keywords)]
        if not target_values:
            print(f"  [Warning] No matching HGD events in {Path(file_path).name}: {list(event_ids.keys())}")
            return None, None
        event_mask = np.isin(events[:, 2], target_values)
        filtered_events = events[event_mask]
        if len(filtered_events) == 0:
            return None, None
        event_id = {str(value): value for value in sorted(set(filtered_events[:, 2]))}
        epochs = mne.Epochs(
            raw,
            filtered_events,
            event_id=event_id,
            tmin=self.tmin,
            tmax=self.tmax - (1.0 / self.target_fs),
            baseline=None,
            picks=picks,
            preload=True,
            verbose=False,
        )
        x_data = epochs.get_data() * 1e6
        labels_raw = epochs.events[:, -1]
        mapping = {old: new for new, old in enumerate(sorted(np.unique(labels_raw)))}
        y_labels = np.array([mapping[label] for label in labels_raw], dtype=np.int64)
        expected_points = int(4.5 * self.target_fs)
        if x_data.shape[2] > expected_points:
            x_data = x_data[:, :, :expected_points]
        return x_data, y_labels

    def build_dataset(self):
        self.save_root.mkdir(parents=True, exist_ok=True)
        for sub in range(1, 15):
            print(f"\nProcessing HGD subject {sub}...")
            train_file = self.root_path / "train" / f"{sub}.edf"
            test_file = self.root_path / "test" / f"{sub}.edf"
            if not train_file.exists() or not test_file.exists():
                print(f"  Missing HGD files for subject {sub}; skipping.")
                continue
            train_x, train_y = self.get_epochs_from_edf(train_file)
            test_x, test_y = self.get_epochs_from_edf(test_file)
            if train_x is None or test_x is None:
                continue
            train_x = euclidean_alignment(train_x)
            test_x = euclidean_alignment(test_x)
            Load_IV2a._save_subject(self, sub, train_x, train_y, test_x, test_y)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build processed EEG datasets for D-STNet.")
    parser.add_argument("--dataset", choices=["2a", "2b", "hgd"], required=True)
    parser.add_argument("--raw_root", type=str, default=None)
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--no_ea", action="store_true", help="Skip per-subject Euclidean Alignment (for learnable-alpha experiment)")
    args = parser.parse_args()

    loaders = {"2a": Load_IV2a, "2b": Load_IV2b, "hgd": Load_HGD}
    if args.dataset == "2a":
        loaders["2a"](raw_root=args.raw_root, save_root=args.save_root, no_ea=args.no_ea).build_dataset()
    else:
        loaders[args.dataset](raw_root=args.raw_root, save_root=args.save_root).build_dataset()
