"""Strict LOSO (leave-one-subject-out) cross-subject evaluation for D-STNet on BCI IV-2a.

Lexicon / why this script exists
--------------------------------
- The shipped ``main.py`` has several integration bugs: it loads the model from
  ``Model/D-STNet.py`` (actual: ``Models/D_STNet.py``) and calls
  ``Net(input_channels=..., dataset=...)`` which ``Net.__init__`` does not accept.
  This script bypasses both by loading the model from the real path and calling it
  with its actual arguments.
- ``main.py``'s LOSO is *leaky*: it selects the best checkpoint by the *test fold*
  accuracy. This script splits the 8 source subjects into 80/20 (stratified) and
  selects the model purely on the validation split, so the left-out subject's labels
  are touched only at the final evaluation. That is the "strict" part.

Protocol (matches eeg-mibci-benchmark):
  dataset: BCI Competition IV-2a (4-class MI, 22 ch, 250 Hz, 9 subjects, LOSO)
  fold:    subject ``s`` is held out -> source pool = the other 8 subjects (2 sessions each)
           source pool stratified-split 80/20 -> train / val (model selection uses val ONLY)
           final evaluation on subject ``s`` (both sessions)
  report:  per-subject acc / kappa / recall / auc, then mean +/- std across subjects

Run (conda env ``DST``):
    PYTHONUNBUFFERED=1 ~/.conda/envs/DST/bin/python run_loso.py \
        --data_raw /home/u24600002150531/8410/data/BCI/2a --seed 42
"""

import argparse
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from sklearn.metrics import cohen_kappa_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

try:
    from lightning.pytorch import seed_everything as lightning_seed_everything
except ImportError:
    lightning_seed_everything = None


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DEFAULTS = {
    "2a": {
        "data_dir_name": "BCIC_2a",
        "num_subjects": 9,
        "num_classes": 4,
        "input_channels": 22,
        "class_names": ["Left", "Right", "Foot", "Tongue"],
    },
}


# ---------------------------------------------------------------------------
# module loading
# ---------------------------------------------------------------------------
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data_module = load_module_from_path("dstnet_data", PROJECT_ROOT / "DataLoader" / "LoadData.py")
model_module = load_module_from_path("dstnet_model", PROJECT_ROOT / "Models" / "D_STNet.py")
Net = model_module.Net


# ---------------------------------------------------------------------------
# seeding / deterministic helpers
# ---------------------------------------------------------------------------
def seed_all(seed):
    # [REPRODUCIBILITY] Keep EXACTLY the settings used for the (iii) 0.7357 baseline:
    #   CUBLAS :16:8 + no cudnn deterministic/benchmark override + use_deterministic_algorithms(False).
    #   (Stronger determinism flags CHANGE the training arithmetic -> different result -> not 0.7357.)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if lightning_seed_everything is not None:
        lightning_seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(False)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# loss / augmentation helpers (mirrors main.py)
# ---------------------------------------------------------------------------
class EEGDataAugmentation:
    def __init__(self, time_noise_level=0.01, prob=0.5):
        self.time_noise_level = float(time_noise_level)
        self.prob = float(prob)

    def __call__(self, x):
        if self.prob <= 0 or torch.rand(1, device=x.device) > self.prob:
            return x
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return x + torch.randn_like(x) * std * self.time_noise_level


def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def unpack_model_output(output):
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


@torch.no_grad()
def evaluate_model(net, loader, criterion, device, num_classes):
    net.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).flatten()
        logits, _ = unpack_model_output(net(x))
        if criterion is not None:
            total_loss += criterion(logits, y).item() * x.size(0)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)
    avg_loss = total_loss / len(loader.dataset) if criterion is not None else 0.0
    acc = float(np.mean(y_pred == y_true))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    try:
        if num_classes > 2:
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            auc_value = roc_auc_score(y_true_bin, y_probs, multi_class="ovr", average="macro")
        else:
            auc_value = roc_auc_score(y_true, y_probs[:, 1])
    except ValueError:
        auc_value = 0.0
    metrics = {"loss": float(avg_loss), "acc": acc, "kappa": kappa, "recall": recall, "auc": float(auc_value)}
    return metrics, y_true, y_pred, y_probs


# ---------------------------------------------------------------------------
# raw -> processed cache
# ---------------------------------------------------------------------------
def build_processed_cache(raw_root, save_root, time_points=1125, no_ea=False):
    """Reuse Load_IV2a preprocessing, handling the split gdf/ + mat/ layout."""
    raw_root = Path(raw_root)
    mat_dir = raw_root / "mat"
    gdf_dir = raw_root / "gdf"
    if not mat_dir.exists():
        mat_dir = raw_root  # fallback: eval .mat may sit alongside .gdf
    if not gdf_dir.exists():
        gdf_dir = raw_root

    loader = data_module.Load_IV2a(raw_root=gdf_dir, save_root=save_root, no_ea=no_ea)
    for sub in range(1, DATASET_DEFAULTS["2a"]["num_subjects"] + 1):
        train_dir = Path(save_root) / f"S{sub}train"
        test_dir = Path(save_root) / f"S{sub}test"
        mpath = train_dir / "data.mat"
        epath = test_dir / "data.mat"
        if mpath.exists() and epath.exists():
            print(f"  [cache] subject {sub} already built, skipping.")
            continue

        print(f"  [build] raw subject {sub}...")
        tr = loader.get_epochs(file_path=gdf_dir / f"A0{sub}T.gdf")
        te = loader.get_session2(
            file_path=gdf_dir / f"A0{sub}E.gdf",
            label_path=mat_dir / f"A0{sub}E.mat",
        )
        train_x = np.asarray(tr["x_data"])[:, :, :time_points]
        test_x = np.asarray(te["x_data"])[:, :, :time_points]
        if not no_ea:
            train_x = data_module.euclidean_alignment(train_x)
            test_x = data_module.euclidean_alignment(test_x)
        train_y = np.asarray(tr["y_labels"]).reshape(-1)
        test_y = np.asarray(te["y_labels"]).reshape(-1)
        loader._save_subject(sub, train_x, train_y, test_x, test_y)
    print(f"  [cache] all subjects saved to {save_root}")


def get_subject_all_arrays(subject, data_path):
    """Both sessions of one subject as (x, y) numpy arrays."""
    import scipy.io as sio

    def load(split):
        mat_path = Path(data_path) / f"S{subject}{split}" / "data.mat"
        mat = sio.loadmat(str(mat_path))
        x = np.asarray(mat["x_data"], dtype=np.float32)
        y = np.asarray(mat["y_data"]).reshape(-1).astype(np.int64)
        return x, y

    tx, ty = load("train")
    ex, ey = load("test")
    return np.concatenate([tx, ex], axis=0), np.concatenate([ty, ey], axis=0)


def concat_subjects_arrays(subjects, data_path):
    xs, ys = [], []
    for s in subjects:
        x, y = get_subject_all_arrays(s, data_path)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------------------------------------------------------
# strict LOSO runner
# ---------------------------------------------------------------------------
class StrictLOSO:
    def __init__(self, configs):
        self.configs = configs
        self.num_subjects = configs["num_subjects"]
        self.num_classes = configs["num_classes"]
        self.input_channels = configs["input_channels"]
        self.device = configs["device"]
        self.epochs = configs["epochs"]
        self.batch_size = configs["batch_size"]
        self.data_path = Path(configs["data_processed"]).resolve()
        self.class_names = configs["class_names"]

    def _make_loader(self, arrays, idx, shuffle):
        x, y = arrays
        ds = data_module.EEGDataset(torch.from_numpy(x[idx]), torch.from_numpy(y[idx]))
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            num_workers=self.configs["num_workers"],
            drop_last=shuffle,
        )

    def _build_model(self):
        net = Net(
            num_classes=self.num_classes,
            input_time_length=self.configs["time_points"],
            fs=self.configs["fs"],
            ea_momentum=self.configs["ea_momentum"],
            learn_alpha=bool(self.configs.get("learn_alpha", False)),
            whiten_train=bool(self.configs.get("whiten_train", False)),
            attn_gate=self.configs.get("attn_gate", "none"),
            spec_mode=self.configs.get("spec_mode", "none"),
        )
        return net.to(self.device)

    def train_subject(self, test_subject):
        cfg = self.configs
        seed = cfg["seed"]
        results_dir = Path(cfg["results_dir"])
        sub_dir = results_dir / f"sub{test_subject}"
        weight_dir = sub_dir / "weight"
        weight_dir.mkdir(parents=True, exist_ok=True)

        # ---- build strict, leak-free folds (validation ONLY from source subjects) ----
        src_subjects = [s for s in range(1, self.num_subjects + 1) if s != test_subject]
        src_x, src_y = concat_subjects_arrays(src_subjects, self.data_path)
        idx_all = np.arange(len(src_y))
        train_idx, val_idx = train_test_split(
            idx_all,
            test_size=cfg["val_ratio"],
            random_state=seed,
            stratify=src_y,
        )
        test_x, test_y = get_subject_all_arrays(test_subject, self.data_path)

        train_loader = self._make_loader((src_x, src_y), train_idx, shuffle=True)
        val_loader = self._make_loader((src_x, src_y), val_idx, shuffle=False)
        test_loader = self._make_loader((test_x, test_y), np.arange(len(test_y)), shuffle=False)

        net = self._build_model()
        criterion_ce = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"]).to(self.device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=cfg["min_lr"])
        augmentor = EEGDataAugmentation(time_noise_level=cfg["time_noise_level"], prob=cfg["aug_prob"])

        best_val_acc = -1.0
        best_path = weight_dir / "best_model.pth"
        history = {"train_loss": [], "val_acc": [], "val_kappa": []}

        print(f"\n>>> LOSO subject {test_subject} | source={len(train_idx)} val={len(val_idx)} test={len(test_y)}")

        for epoch in range(self.epochs):
            net.train()
            total_loss, total = 0.0, 0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).flatten()
                inputs = augmentor(inputs)
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, cfg["mixup_alpha"])
                optimizer.zero_grad(set_to_none=True)
                logits, _ = unpack_model_output(net(inputs))
                loss = mixup_criterion(criterion_ce, logits, targets_a, targets_b, lam)
                loss.backward()
                if cfg["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
                total += labels.size(0)
            scheduler.step()

            val_metrics, _, _, _ = evaluate_model(net, val_loader, criterion_ce, self.device, self.num_classes)
            history["train_loss"].append(total_loss / max(total, 1))
            history["val_acc"].append(val_metrics["acc"])
            history["val_kappa"].append(val_metrics["kappa"])

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                torch.save(net.state_dict(), best_path)

            if epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1:
                print(
                    f"  Ep:{epoch:03d} | Train L:{total_loss / max(total, 1):.4f} | "
                    f"Val A:{val_metrics['acc']:.4f} K:{val_metrics['kappa']:.4f} | BestValA:{best_val_acc:.4f}"
                )

        # ---- evaluate the best-val model on the held-out test subject ONLY now ----
        net.load_state_dict(torch.load(best_path, map_location=self.device))
        test_metrics, y_true, y_pred, _ = evaluate_model(net, test_loader, criterion_ce, self.device, self.num_classes)
        best_model_val = evaluate_model(net, val_loader, criterion_ce, self.device, self.num_classes)[0]

        np.save(sub_dir / "confusion_true.npy", y_true)
        np.save(sub_dir / "confusion_pred.npy", y_pred)
        with open(sub_dir / "result.txt", "w", encoding="utf-8") as f:
            f.write(f"Subject {test_subject} strict-LOSO result (model selected by validation)\n")
            f.write(f"best_val_acc: {best_model_val['acc']:.6f}\n")
            for k, v in test_metrics.items():
                f.write(f"test_{k}: {v:.6f}\n")

        return {
            "subject": test_subject,
            "best_val_acc": float(best_model_val["acc"]),
            "test": test_metrics,
        }

    def run(self):
        results_dir = Path(self.configs["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        fold_results = []
        for subject in range(1, self.num_subjects + 1):
            fold_results.append(self.train_subject(subject))

        per_subject = {
            "subjects": [r["subject"] for r in fold_results],
            "best_val_accs": [r["best_val_acc"] for r in fold_results],
            "test_acc": [r["test"]["acc"] for r in fold_results],
            "test_kappa": [r["test"]["kappa"] for r in fold_results],
            "test_recall": [r["test"]["recall"] for r in fold_results],
            "test_auc": [r["test"]["auc"] for r in fold_results],
        }
        acc = np.array(per_subject["test_acc"])
        kappa = np.array(per_subject["test_kappa"])
        summary = {
            "protocol": "strict_loso_20pct_source_val",
            "seed": self.configs["seed"],
            "n_folds": self.num_subjects,
            "per_subject": per_subject,
            "acc_mean": float(acc.mean()),
            "acc_std": float(acc.std(ddof=1)),
            "kappa_mean": float(kappa.mean()),
            "kappa_std": float(kappa.std(ddof=1)),
            "acc_min": float(acc.min()),
            "acc_max": float(acc.max()),
            "report": f"acc = {acc.mean():.4f} +/- {acc.std(ddof=1):.4f} | kappa = {kappa.mean():.4f} +/- {kappa.std(ddof=1):.4f}",
        }
        with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print(f"Strict LOSO D-STNet on BCI IV-2a (seed={self.configs['seed']})")
        print(f"  per-subject acc: {[f'{a:.4f}' for a in per_subject['test_acc']]}")
        print(f"  acc = {acc.mean():.4f} +/- {acc.std(ddof=1):.4f}")
        print(f"  kappa = {kappa.mean():.4f} +/- {kappa.std(ddof=1):.4f}")
        print(f"  results -> {results_dir}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# config / entry
# ---------------------------------------------------------------------------
def build_config(args):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    defaults = DATASET_DEFAULTS["2a"]
    return vars(args) | {
        "dataset_key": "2a",
        "num_subjects": defaults["num_subjects"],
        "num_classes": defaults["num_classes"],
        "input_channels": defaults["input_channels"],
        "class_names": defaults["class_names"],
        "device": device,
    }


def main():
    parser = argparse.ArgumentParser(description="Strict LOSO D-STNet on BCI IV-2a (leak-free).")
    parser.add_argument("--data_raw", type=str, default="/home/u24600002150531/8410/data/BCI/2a")
    parser.add_argument("--data_processed", type=str, default=None,
                        help="Processed cache dir (S{s}train|test/data.mat). Default: <data_raw/processed/BCIC_2a>.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--time_points", type=int, default=1125)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--ea_momentum", type=float, default=0.9)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--time_noise_level", type=float, default=0.01)
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default=None, help="Override results dir.")
    parser.add_argument("--build_only", action="store_true", help="Build the processed cache and exit.")
    parser.add_argument("--learn_alpha", type=int, default=0, help="learnable data-adaptive alpha in DynamicEA")
    parser.add_argument("--whiten_train", type=int, default=0, help="apply dynamic whitening during training (needed to learn alpha)")
    parser.add_argument("--no_ea", type=int, default=0, help="build/use the no-static-EA processed cache")
    parser.add_argument("--attn_gate", choices=["elementwise", "headwise", "none", "content"], default="none",
                        help="Attention gating mode (none=default/no gate [validated best], elementwise=original)")
    parser.add_argument("--spec_mode", choices=["none", "C", "B", "D"], default="none",
                        help="Spectral/GASL improvement: none=fixed 5 bands / C=band weights / B=band attention / D=learnable Gaussian filterbank")
    args = parser.parse_args()

    raw_root = Path(args.data_raw).resolve()
    no_ea = bool(args.no_ea)
    processed = Path(args.data_processed).resolve() if args.data_processed else (
        raw_root / "processed" / ("BCIC_2a_noea" if no_ea else "BCIC_2a")
    )
    processed.mkdir(parents=True, exist_ok=True)

    print(f"[env] python | torch {torch.__version__} | cuda {torch.cuda.is_available()} | device will be {args.device}")
    print(f"[data] raw={raw_root}\n[data] processed={processed}  no_ea={no_ea}")

    # 1) ensure processed cache
    build_processed_cache(raw_root, processed, time_points=args.time_points, no_ea=no_ea)
    if args.build_only:
        return

    # 2) config + results dir
    configs = build_config(args)
    configs["data_processed"] = str(processed)
    configs["results_dir"] = str(
        Path(args.out_dir).resolve()
        if args.out_dir
        else PROJECT_ROOT / "results" / f"loso_2a_seed{args.seed}_{time.strftime('%Y-%m-%d--%H-%M', time.localtime())}"
    )
    configs["data_raw"] = str(raw_root)

    env_info = {
        "python": str(Path(os.__file__).resolve()),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": configs["device"],
    }
    configs["env"] = env_info

    results_dir = Path(configs["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)

    seed_all(args.seed)
    configs["data_processed"] = str(processed)
    StrictLOSO(configs).run()




if __name__ == "__main__":
    main()
