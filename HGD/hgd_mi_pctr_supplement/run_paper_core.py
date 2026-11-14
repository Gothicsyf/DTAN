"""D-STNet SSDA with source-only class-feature-prototype TAT on BCI IV-2a.

Protocol (strict, no leakage):
  fold (test subject = s):
    source       : the other 8 subjects, both sessions, fully labeled
    target       : subject s
      labeled    : k trials/class sampled from s's SESSION-T (train) -> 4k labeled trials
      unlabeled  : the remaining trials of s's SESSION-T      -> entropy-minimisation pool
      TEST       : s's SESSION-E (held-out, never used in training / model selection)
  Adapt: load a source-pretrained backbone, fine-tune on (source labeled + target labeled CE)
         plus lambda_ent * entropy-minimisation on target unlabeled, and (optional) anchor
         the SD-EA reference covariance to the target-labeled statistics.
  Model selection: source backbone picked by an internal source validation split;
         adaptation uses a fixed epoch budget (no selection on session-E).

K-sweep: k in {0,1,2,4,8} per class -> "accuracy vs label budget" curve (k=0 == pure DA).

Run:
    PYTHONUNBUFFERED=1 ~/.conda/envs/DST/bin/python run_ssda_prototype_tat.py \
        --data_raw /home/u24600002150531/8410/data/BCI/2a --ks 0 --tat_weight 0.03
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import run_loso as base

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET = {"num_subjects":14,"num_classes":4,"input_channels":44,"class_names":["Left","Right","Feet","Rest"]}

# Motor-effector topology, class order: Left Hand, Right Hand, Feet, Rest. Rest occupies the former tongue slot by explicit user instruction; numeric matrix unchanged.
# This is the same normalized T used by experiments/shared/scripts/final_tat.py.
from pctr import prototype_tat_loss as waveform_loss
from pctr_fold import fold_prior


def load_sub_session(subject, session, data_path):
    """Load one session of one processed subject. session: 'train'->T, 'test'->E."""
    m = sio.loadmat(str(Path(data_path) / f"S{subject}{session}" / "data.mat"))
    return np.asarray(m["x_data"], dtype=np.float32), np.asarray(m["y_data"]).reshape(-1).astype(np.int64)


def concat_source(subjects, data_path):
    xs, ys = [], []
    for s in subjects:
        for sess in ("train", "test"):
            x, y = load_sub_session(s, sess, data_path)
            xs.append(x)
            ys.append(y)
    return np.concatenate(xs, 0), np.concatenate(ys, 0)


def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    return -(probs * (probs + 1e-12).log()).sum(dim=1).mean()


def topology_for_mode(mode, seed=42):
    """Return a pre-declared topology while preserving the original TAT scale."""
    if mode == "correct":
        return TAT_TOPOLOGY.clone()
    if mode == "permuted":
        # Fixed non-identity class permutation, declared before any result is seen.
        order = torch.tensor([0, 2, 1, 3])
        return TAT_TOPOLOGY[order][:, order].clone()
    if mode == "uniform":
        out = torch.ones_like(TAT_TOPOLOGY)
        out.fill_diagonal_(0.0)
        return out
    if mode == "random":
        rng = np.random.RandomState(seed)
        values = TAT_TOPOLOGY.numpy()[np.triu_indices(4, 1)].copy()
        rng.shuffle(values)
        out = np.zeros((4, 4), dtype=np.float32)
        iu = np.triu_indices(4, 1)
        out[iu] = values
        out[(iu[1], iu[0])] = values
        return torch.from_numpy(out)
    raise ValueError(f"Unknown topology mode: {mode}")


def prototype_tat_loss(features, labels, topology, num_classes=4):
    return waveform_loss(features, labels, topology, num_classes)


def prototype_geometry(net, loader, device, topology, num_classes=4):
    """Full-source prototype MSE/rho after adaptation, for diagnostics only."""
    from scipy.stats import spearmanr

    was_training = net.training
    net.eval()
    sums = None
    counts = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).flatten()
            _, features = base.unpack_model_output(net(xb))
            if sums is None:
                sums = torch.zeros(num_classes, features.shape[1], device=device, dtype=features.dtype)
            sums.index_add_(0, yb, features)
            counts.index_add_(0, yb, torch.ones_like(yb, dtype=counts.dtype))
        prototypes = sums / counts.clamp_min(1).unsqueeze(1)
        distances = torch.cdist(prototypes, prototypes)
        normalized = distances / (distances.max() + 1e-8)
        target = topology.to(device=normalized.device, dtype=normalized.dtype)
        mse = F.mse_loss(normalized, target).item()
        iu = np.triu_indices(target.shape[0], 1)
        rho = spearmanr(normalized.detach().cpu().numpy()[iu], target.cpu().numpy()[iu]).statistic
    net.train(was_training)
    return float(mse), float(rho)


def stratified_labeled_indices(y, k, seed, num_classes=4):
    """Sample k trials per class -> absolute indices (deterministic per (seed, subject, k))."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(num_classes):
        cand = np.where(y == c)[0]
        rng.shuffle(cand)
        idx.append(cand[:k])
    return np.concatenate(idx) if idx else np.array([], dtype=int)


class SSDA:
    def __init__(self, configs):
        self.cfg = configs
        self.device = configs["device"]
        self.ks = configs["ks"]
        self.num_classes = DATASET["num_classes"]
        self.data_path = Path(configs["data_processed"]).resolve()
        self.topology = None  # initialized for each held-out subject

    def build_model(self):
        net = base.Net(
            num_classes=self.num_classes,
            input_time_length=self.cfg["time_points"],
            fs=self.cfg["fs"],
            ea_momentum=self.cfg["ea_momentum"],
            use_dynamic_ea=bool(self.cfg.get("use_dynamic_ea", False)),
            learn_alpha=bool(self.cfg.get("learn_alpha", False)),
            whiten_train=bool(self.cfg.get("whiten_train", False)),
            ea_mode=self.cfg.get("ea_mode", "whiten"),
            use_feature_align=bool(self.cfg.get("use_feature_align", False)),
            fa_learn_alpha=bool(self.cfg.get("fa_learn_alpha", True)),
            attn_gate=self.cfg.get("attn_gate", "none"),
            spec_mode=self.cfg.get("spec_mode", "none"),
        )
        return net.to(self.device)

    # ---------------- source pre-training ----------------
    def pretrain_source(self, src_x, src_y):
        cfg = self.cfg
        idx = np.arange(len(src_y))
        tr_idx, val_idx = train_test_split(idx, test_size=cfg["val_ratio"], random_state=cfg["seed"], stratify=src_y)
        train_loader = self._sup_loader((src_x, src_y), tr_idx)
        val_loader = self._sup_loader((src_x, src_y), val_idx, shuffle=False)

        net = self.build_model()
        crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"]).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = CosineAnnealingLR(opt, T_max=cfg["pt_epochs"], eta_min=cfg["min_lr"])
        aug = base.EEGDataAugmentation(time_noise_level=cfg["time_noise_level"], prob=cfg["aug_prob"])

        best_val, best_state = -1.0, None
        pbar = tqdm(range(cfg["pt_epochs"]), desc="pre-train", leave=False)
        for epoch in pbar:
            net.train()
            for xb, yb in train_loader:
                xb = aug(xb.to(self.device))
                yb = yb.to(self.device).flatten()
                mixed_xb, ya, yb2, lam = base.mixup_data(xb, yb, cfg["mixup_alpha"])
                opt.zero_grad(set_to_none=True)
                logits, _ = base.unpack_model_output(net(mixed_xb))
                loss = base.mixup_criterion(crit, logits, ya, yb2, lam)
                # Preserve the RNG stream used by the baseline: this extra
                # prototype forward consumes Dropout randomness locally only.
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                _, prototype_features = base.unpack_model_output(net(xb))
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                loss = loss + cfg["tat_weight"] * prototype_tat_loss(
                    prototype_features, yb, self.topology, self.num_classes
                )
                loss.backward()
                opt.step()
            sched.step()
            vm, _, _, _ = base.evaluate_model(net, val_loader, crit, self.device, self.num_classes)
            pbar.set_postfix(val_acc=f"{vm['acc']:.4f}", best=f"{best_val:.4f}")
            if vm["acc"] > best_val:
                best_val = vm["acc"]
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
        pbar.close()
        net.load_state_dict(best_state)
        return net

    def save_model(self, net, sub_dir):
        sub_dir.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), sub_dir / "model.pth")

    # ---------------- adaptation -----------------
    def anchor_reference_cov(self, net, lbl_x):
        """Method-D: seed the SD-EA reference covariance from target-labeled statistics."""
        if not self.cfg["use_labeled_cov"] or lbl_x.shape[0] < 2:
            return
        x = torch.from_numpy(lbl_x).float().to(self.device)
        xc = x - x.mean(dim=2, keepdim=True)
        covs = torch.bmm(xc, xc.transpose(1, 2)) / (x.shape[2] - 1)
        net.dynamic_ea.running_cov.copy_(covs.mean(0))
        net.dynamic_ea.is_initialized = True

    @staticmethod
    def configure_adaptation_parameters(net, scope):
        """Freeze all but the parameter subset declared by the R042 ablation."""
        valid = {"all", "head_fa", "norm_fa", "fa_only"}
        if scope not in valid:
            raise ValueError(f"Unknown update_scope={scope!r}; expected one of {sorted(valid)}")

        for parameter in net.parameters():
            parameter.requires_grad_(scope == "all")

        if scope in {"head_fa", "fa_only"}:
            prefixes = ("feature_align.",) if scope == "fa_only" else ("classifier_head.", "feature_align.")
            for name, parameter in net.named_parameters():
                if name.startswith(prefixes):
                    parameter.requires_grad_(True)
        elif scope == "norm_fa":
            norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                          nn.InstanceNorm2d, nn.InstanceNorm3d)
            norm_names = {
                f"{module_name}.{parameter_name}" if module_name else parameter_name
                for module_name, module in net.named_modules()
                if isinstance(module, norm_types)
                for parameter_name, _ in module.named_parameters(recurse=False)
            }
            for name, parameter in net.named_parameters():
                if name in norm_names or name.startswith("feature_align."):
                    parameter.requires_grad_(True)

        trainable = [parameter for parameter in net.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError(f"update_scope={scope!r} selected no trainable parameters")
        return trainable, sum(p.numel() for p in trainable), sum(p.numel() for p in net.parameters())

    def adapt_and_eval(self, net, test_subject, k, src_x, src_y):
        cfg = self.cfg
        # target session-T (labeled pool) and session-E (test)
        t_x, t_y = load_sub_session(test_subject, "train", self.data_path)
        e_x, e_y = load_sub_session(test_subject, "test", self.data_path)

        lbl_idx = stratified_labeled_indices(t_y, k, seed=cfg["seed"] + test_subject * 100 + k, num_classes=self.num_classes)
        lbl_x, lbl_y = t_x[lbl_idx], t_y[lbl_idx]
        unl_x = t_x[np.delete(np.arange(len(t_y)), lbl_idx)]
        if cfg["adapt_mode"] == "wrong_target":
            wrong_subject = test_subject % DATASET["num_subjects"] + 1
            wrong_x, _ = load_sub_session(wrong_subject, "train", self.data_path)
            unl_x = wrong_x

        # Deployment-budget ablations use a deterministic chronological prefix
        # of the target Session-T pool.  The default 1.0 preserves all legacy
        # runs exactly.  A zero-target endpoint is represented by the existing
        # source-replay-only experiment rather than constructing an empty loader.
        target_fraction = float(cfg.get("target_fraction", 1.0))
        if not 0.0 < target_fraction <= 1.0:
            raise ValueError("target_fraction must be in (0, 1]; use R011 for the 0% endpoint")
        target_count = max(1, int(np.ceil(len(unl_x) * target_fraction)))
        unl_x = unl_x[:target_count]

        # Deterministic stratified source replay budget.  This affects only the
        # adaptation stage; source pretraining/checkpoints remain identical.
        source_fraction = float(cfg.get("source_fraction", 1.0))
        if not 0.0 < source_fraction <= 1.0:
            raise ValueError("source_fraction must be in (0, 1]")
        if source_fraction < 1.0:
            keep = []
            rng = np.random.RandomState(cfg["seed"] + test_subject * 1000 + 17)
            for cls in range(self.num_classes):
                cls_idx = np.flatnonzero(src_y == cls)
                rng.shuffle(cls_idx)
                keep.extend(cls_idx[:max(1, int(np.ceil(len(cls_idx) * source_fraction)))])
            keep = np.asarray(sorted(keep), dtype=np.int64)
            src_x_adapt, src_y_adapt = src_x[keep], src_y[keep]
        else:
            src_x_adapt, src_y_adapt = src_x, src_y

        self.anchor_reference_cov(net, lbl_x)

        sup = self._sup_loader((np.concatenate([src_x_adapt, lbl_x], 0), np.concatenate([src_y_adapt, lbl_y], 0)),
                               np.arange(len(src_y_adapt) + len(lbl_y)))
        unl_loader = self._unl_loader(unl_x)

        crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"]).to(self.device)
        trainable, trainable_count, total_count = self.configure_adaptation_parameters(
            net, cfg.get("update_scope", "all")
        )
        print(f"update_scope={cfg.get('update_scope', 'all')} "
              f"trainable={trainable_count}/{total_count}", flush=True)
        opt = torch.optim.AdamW(trainable, lr=cfg["adapt_lr"], weight_decay=cfg["weight_decay"])
        sched = CosineAnnealingLR(opt, T_max=cfg["adapt_epochs"], eta_min=cfg["min_lr"])
        aug = base.EEGDataAugmentation(time_noise_level=cfg["time_noise_level"], prob=cfg["aug_prob"])

        net.train()
        pbar = tqdm(range(cfg["adapt_epochs"]), desc=f"adapt S{test_subject} k={k}", leave=False)
        for epoch in pbar:
            unl_iter = iter(unl_loader)
            for xb, yb in sup:
                xb = aug(xb.to(self.device))
                yb = yb.to(self.device).flatten()
                xb, ya, yb2, lam = base.mixup_data(xb, yb, cfg["mixup_alpha"])
                opt.zero_grad(set_to_none=True)
                logits, _ = base.unpack_model_output(net(xb))
                loss = base.mixup_criterion(crit, logits, ya, yb2, lam)
                if cfg["lambda_ent"] > 0 and len(unl_loader) > 0:
                    try:
                        xu = next(unl_iter)[0]
                    except StopIteration:
                        unl_iter = iter(unl_loader)
                        xu = next(unl_iter)[0]
                    xu = aug(xu.to(self.device))
                    logits_u, _ = base.unpack_model_output(net(xu))
                    loss = loss + cfg["lambda_ent"] * entropy_loss(logits_u)
                # Source-only TAT: no fixed, target-agnostic topology penalty
                # during k=0 subject adaptation.
                loss.backward()
                opt.step()
            sched.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()

        # evaluate on held-out session-E (strict: only touched here)
        e_loader = self._sup_loader((e_x, e_y), np.arange(len(e_y)), shuffle=False)
        metrics, y_true, y_pred, _ = base.evaluate_model(net, e_loader, crit, self.device, self.num_classes)
        metrics["adapt_trainable_params"] = float(trainable_count)
        metrics["adapt_total_params"] = float(total_count)
        source_eval = self._sup_loader((src_x, src_y), np.arange(len(src_y)), shuffle=False)
        metrics["tat_mse"], metrics["tat_rho"] = prototype_geometry(
            net, source_eval, self.device, self.topology, self.num_classes
        )
        return metrics, y_true, y_pred

    def _sup_loader(self, arrays, idx, shuffle=True):
        x, y = arrays
        ds = base.data_module.EEGDataset(torch.from_numpy(x[idx]), torch.from_numpy(y[idx]))
        return DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=shuffle,
                          pin_memory=torch.cuda.is_available(), num_workers=self.cfg["num_workers"],
                          worker_init_fn=base.seed_worker, drop_last=shuffle)

    def _unl_loader(self, x):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(x))
        return DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=True,
                          pin_memory=torch.cuda.is_available(), num_workers=self.cfg["num_workers"],
                          worker_init_fn=base.seed_worker, drop_last=True)

    # ---------------- orchestration ----------------
    def run(self):
        cfg = self.cfg
        out_root = Path(cfg["results_dir"])
        subjects = cfg["subjects"]
        for subject in subjects:
            src_subjects = [s for s in range(1, DATASET["num_subjects"] + 1) if s != subject]
            src_x, src_y = concat_source(src_subjects, self.data_path)
            global TAT_TOPOLOGY
            TAT_TOPOLOGY = fold_prior(self.data_path, src_subjects, src_y, out_root / f"subject{subject}")
            self.topology = topology_for_mode(cfg["topology_mode"], cfg["topology_seed"])
            sub_root = out_root / f"subject{subject}"
            sub_root.mkdir(parents=True, exist_ok=True)
            source_pth = sub_root / "source_model.pth"
            if cfg.get("source_checkpoint_root"):
                source_pth = Path(str(cfg["source_checkpoint_root"]).removesuffix("_waveform_max") + "_waveform_max") / f"subject{subject}" / "source_model.pth"

            # resume: reuse a pretrained source backbone if present
            if source_pth.exists():
                print(f"\n>>> SSDA subject {subject}: reusing pretrained source ...")
                net = self.build_model()
                net.load_state_dict(torch.load(source_pth, map_location=self.device))
            else:
                print(f"\n>>> SSDA subject {subject}: pretraining source ({len(src_subjects)} subjects) ...")
                net = self.pretrain_source(src_x, src_y)
                source_pth.parent.mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), source_pth)

            for k in cfg["ks"]:
                sub_dir = sub_root / f"k{k}"
                if (sub_dir / "result.txt").exists():
                    print(f"    k={k} already done, skipping", flush=True)
                    continue
                t_start = time.time()
                print(f"    k={k} ... ", end="", flush=True)
                # each k is independent: reset to the SAME pretrained source backbone
                net = self.build_model()
                net.load_state_dict(torch.load(source_pth, map_location=self.device))
                metrics, y_true, y_pred = self.adapt_and_eval(net, subject, k, src_x, src_y)
                sub_dir.mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), sub_dir / "model.pth")
                np.save(sub_dir / "confusion_true.npy", y_true)
                np.save(sub_dir / "confusion_pred.npy", y_pred)
                with open(sub_dir / "result.txt", "w", encoding="utf-8") as f:
                    f.write(f"Subject {subject} SSDA k={k} result\n")
                    for key, val in metrics.items():
                        f.write(f"{key}: {val:.6f}\n")
                print(f"acc={metrics['acc']:.4f} kappa={metrics['kappa']:.4f} ({time.time() - t_start:.0f}s)", flush=True)

        self.write_summary(subjects)

    def write_summary(self, subjects):
        cfg = self.cfg
        out_root = Path(cfg["results_dir"])
        per_k = {}
        for k in cfg["ks"]:
            accs, kap = [], []
            for sub in subjects:
                r = {}
                for line in open(out_root / f"subject{sub}" / f"k{k}" / "result.txt"):
                    if ": " not in line:
                        continue
                    key, val = line.strip().split(": ")
                    r[key] = float(val)
                accs.append(r["acc"])
                kap.append(r["kappa"])
            accs, kap = np.array(accs), np.array(kap)
            per_k[k] = {"subjects": subjects, "acc": accs.tolist(), "kappa": kap.tolist(),
                        "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
                        "kappa_mean": float(kap.mean()), "kappa_std": float(kap.std(ddof=1))}
            print(f"\n  [k={k}] acc = {accs.mean():.4f} +/- {accs.std(ddof=1):.4f} | kappa = {kap.mean():.4f} +/- {kap.std(ddof=1):.4f}")

        with open(out_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"seed": cfg["seed"], "ks": cfg["ks"], "per_k": per_k}, f, indent=2, ensure_ascii=False)
        with open(out_root / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\nSSDA done. -> {out_root}")


def build_config(args):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    return vars(args) | {
        "device": device,
        "num_classes": DATASET["num_classes"],
        "dataset": "HGD", "class_order": ["left_hand","right_hand","feet","rest"],
        "tat_semantics": "unchanged numeric 2a matrix; rest substituted for tongue by user request",
        "data_processed": str(args.data_processed),
        "subjects": args.subjects or list(range(1, DATASET["num_subjects"] + 1)),
    }


def main():
    ap = argparse.ArgumentParser(description="SSDA-LOSO for D-STNet on BCI IV-2a.")
    ap.add_argument("--data_raw", default="/home/u24600002150531/8410/data/BCI/2a")
    ap.add_argument("--data_processed", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--subjects", type=int, nargs="+", default=None)
    ap.add_argument("--pt_epochs", type=int, default=100)
    ap.add_argument("--adapt_epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adapt_lr", type=float, default=5e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--time_points", type=int, default=1125)
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--ea_momentum", type=float, default=0.9)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--mixup_alpha", type=float, default=0.4)
    ap.add_argument("--time_noise_level", type=float, default=0.01)
    ap.add_argument("--aug_prob", type=float, default=0.5)
    ap.add_argument("--lambda_ent", type=float, default=0.3)
    ap.add_argument("--tat_weight", type=float, default=0.03,
                    help="weight of original TAT MSE regularizer on classifier-head geometry")
    ap.add_argument("--topology_mode", choices=["correct", "permuted", "random", "uniform"], default="correct")
    ap.add_argument("--topology_seed", type=int, default=24601)
    ap.add_argument("--adapt_mode", choices=["full", "source_replay_only", "wrong_target"], default="full")
    ap.add_argument("--target_fraction", type=float, default=1.0,
                    help="chronological fraction of target Session-T used unlabeled during adaptation")
    ap.add_argument("--source_fraction", type=float, default=1.0,
                    help="stratified fraction of source trials replayed during adaptation")
    ap.add_argument("--update_scope", choices=["all", "head_fa", "norm_fa", "fa_only"], default="all",
                    help="R042 parameter subset allowed to update during adaptation")
    ap.add_argument("--experiment_id", default="unregistered")
    ap.add_argument("--source_checkpoint_root", default=None,
                    help="optional shared fold-checkpoint root for fair adaptation/ablation reuse")
    ap.add_argument("--use_labeled_cov", type=int, default=1)
    ap.add_argument("--use_dynamic_ea", type=int, default=0, help="enable DynamicEA (SD-EA) in forward")
    ap.add_argument("--learn_alpha", type=int, default=0, help="learnable data-adaptive alpha in DynamicEA")
    ap.add_argument("--whiten_train", type=int, default=0, help="apply dynamic whitening during training (needed to learn alpha)")
    ap.add_argument("--ea_mode", choices=["whiten", "interp"], default="whiten",
                    help="DynamicEA mode: whiten=covariance whitening / interp=alpha interpolation (no whitening)")
    ap.add_argument("--no_ea", type=int, default=0, help="use no-static-EA processed cache")
    ap.add_argument("--use_feature_align", type=int, default=0, help="[Design A] feature-level learnable-alpha align (keep static EA)")
    ap.add_argument("--fa_learn_alpha", type=int, default=1, help="learnable alpha inside FeatureAlign")
    ap.add_argument("--attn_gate", choices=["elementwise", "headwise", "none", "content"], default="none",
                    help="Attention gating mode (none=default/no gate [validated best], elementwise=original, content=improved)")
    ap.add_argument("--spec_mode", choices=["none", "C", "B", "D"], default="none",
                    help="Spectral/GASL improvement: none=fixed 5 bands / C=band weights / B=band attention / D=learnable Gaussian filterbank")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    raw_root = Path(args.data_raw).resolve()
    no_ea = bool(args.no_ea)
    processed = Path(args.data_processed).resolve() if args.data_processed else (
        raw_root / "processed" / ("BCIC_2a_noea" if no_ea else "BCIC_2a")
    )
    if not (processed / "S1train" / "data.mat").exists():
        base.build_processed_cache(raw_root, processed, time_points=args.time_points, no_ea=no_ea)

    cfg = build_config(args)
    cfg["data_processed"] = str(processed)
    cfg["results_dir"] = str(Path(args.out_dir).resolve() if args.out_dir
                             else PROJECT_ROOT / "results" / f"ssda_seed{args.seed}_{time.strftime('%Y-%m-%d--%H-%M', time.localtime())}")
    cfg["results_dir"] = str(Path(cfg["results_dir"]).with_name(Path(cfg["results_dir"]).name + "_waveform_max"))
    cfg["env"] = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "device": cfg["device"]}
    cfg["script"] = str(Path(__file__).resolve())
    cfg["python"] = sys.version
    cfg["platform"] = platform.platform()
    cfg["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    base.seed_all(args.seed)
    SSDA(cfg).run()


if __name__ == "__main__":
    main()
