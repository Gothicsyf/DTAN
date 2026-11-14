"""Paired PhysioNet three-class LOSO with source-validation checkpoint selection.

The two modes differ only in the pre-EA waveform population used for PCTR:
all source rows (including source validation) versus the current training batch. Target
labels never enter training, validation, checkpoint selection, or PCTR.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import nn

from .data import ea_transform, load_aligned, load_all
from .model import Net
from .pctr import prototype_tat_loss, signal_topology

ROOT = Path(__file__).resolve().parents[1]
INTERPRETER = ROOT.parent / 'wrcc_mia_pctr' / '.conda-env' / 'python.exe'
SUBJECTS = tuple(range(1, 11))
CHINA_TZ = timezone(timedelta(hours=8))


def timestamp():
    return datetime.now(CHINA_TZ).isoformat(timespec='seconds')


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)






def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def make_split(y, subject_ids, trial_ids, target, seed):
    """Return source-all/train/validation and disjoint target-adapt/test rows.

    Source validation is trial-stratified within the collection of 9 source
    people. Target stratification uses labels only to fix a reproducible split.
    Target adaptation labels are not returned to the training function.
    """
    y, subject_ids, trial_ids = map(np.asarray, (y, subject_ids, trial_ids))
    if target not in SUBJECTS or set(np.unique(subject_ids)) != set(SUBJECTS):
        raise ValueError('The protocol requires the exact 10-subject inclusion list.')
    if len(y) != len(subject_ids) or len(y) != len(trial_ids) or set(np.unique(y)) != {0, 1, 2}:
        raise ValueError('Invalid aligned labels, subject IDs, or trial IDs.')
    source = np.flatnonzero(subject_ids != target)
    target_rows = np.flatnonzero(subject_ids == target)
    source_train, source_val = train_test_split(
        source, test_size=.2, stratify=y[source], random_state=seed + 1000 * target)
    source_train, source_val = np.sort(source_train), np.sort(source_val)
    rng = np.random.default_rng(seed + 100 * target)
    adapt, test = [], []
    for cls in range(3):
        rows = target_rows[y[target_rows] == cls].copy()
        if len(rows) < 6:
            raise ValueError(f'Subject {target} class {cls} has too few trials.')
        rng.shuffle(rows)
        n_adapt = max(2, len(rows) // 3)
        adapt.extend(rows[:n_adapt].tolist())
        test.extend(rows[n_adapt:].tolist())
    adapt, test = np.sort(np.asarray(adapt, dtype=np.int64)), np.sort(np.asarray(test, dtype=np.int64))
    groups = [set(map(int, rows)) for rows in (source_train, source_val, adapt, test)]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4)):
        raise AssertionError('A trial appears in multiple split partitions.')
    if set(source) != groups[0] | groups[1] or set(target_rows) != groups[2] | groups[3]:
        raise AssertionError('The split does not cover every eligible trial.')
    if target in set(subject_ids[source]) or set(subject_ids[adapt]) != {target} or set(subject_ids[test]) != {target}:
        raise AssertionError('LOSO subject isolation failed.')

    def records(rows):
        return [dict(row=int(row), subject=int(subject_ids[row]), trial=int(trial_ids[row])) for row in rows]

    row_sets = dict(source=source, source_train=source_train, source_validation=source_val,
                    target_adapt=adapt, target_test=test)
    manifest = dict(
        target_subject=int(target), seed=int(seed), source_subjects=sorted(set(subject_ids[source].tolist())),
        selected_subjects=list(SUBJECTS), excluded_subjects=list(range(11, 110)),
        **{name + '_rows': records(rows) for name, rows in row_sets.items()},
        **{name + '_class_counts': np.bincount(y[rows], minlength=3).tolist() for name, rows in row_sets.items()},
        source_validation='80%/20% trial-stratified source split; validation accuracy selects the best epoch; '
                          'lower un-smoothed validation CE breaks accuracy ties, then the earliest epoch wins.',
        source_ea='Per-source-subject unsupervised EA fits all eligible trials of that source person; '
                  'source-validation signals, but never labels, enter this explicitly declared source normalization.',
        pctr_population='Global mode uses all 9 source people, including source_train and source_validation '
                        'labels, before EA/augmentation as explicitly requested; batch mode uses only the '
                        'current source_train batch. Both exclude every target-subject trial.',
        label_access='Source-train labels: pretraining and PCTR; source-validation labels: checkpoint selection '
                     'and, in global mode, the fixed all-source PCTR matrix. '
                     'Source train+validation labels: supervised adaptation. Target labels: fixed stratified '
                     'partition and evaluation after both training stages only.',
        target_protocol='Artificial per-class floor(n/3) unlabeled adaptation split; all remaining trials test. '
                        'This is LOSO with target adaptation, not source-only LOSO and not original sessions.',
        target_ea='Fit only target-adaptation signals and apply the same whitening to target-test signals.',
        adaptation='Start from the source-validation best epoch; train all model parameters with full-source '
                   'supervised CE plus 0.3 target-adaptation entropy, without PCTR; use fixed final adaptation epoch.')
    return source, source_train, source_val, adapt, test, manifest


def batches(n, batch_size, rng=None):
    """Cover every row, merging a singleton tail to satisfy FeatureAlign."""
    if n < 2 or batch_size < 2:
        raise ValueError('FeatureAlign requires at least two samples per batch.')
    order = np.arange(n, dtype=np.int64) if rng is None else rng.permutation(n)
    parts = [order[start:start + batch_size] for start in range(0, n, batch_size)]
    if len(parts) > 1 and len(parts[-1]) == 1:
        parts[-2] = np.concatenate([parts[-2], parts[-1]])
        parts.pop()
    if any(len(part) < 2 for part in parts):
        raise AssertionError('Unmerged singleton batch.')
    return parts


def global_signal_topology(pre, labels, rows, chunk_size=128):
    """Compute exact class means with bounded memory, then use the shared formula."""
    means, counts = [], []
    for cls in range(3):
        class_rows = rows[labels[rows] == cls]
        if not len(class_rows):
            raise ValueError('Global source population must contain all classes.')
        total = np.zeros(pre.shape[1:], dtype=np.float64)
        for start in range(0, len(class_rows), chunk_size):
            block = np.asarray(pre[class_rows[start:start + chunk_size]], dtype=np.float64)
            if not np.isfinite(block).all():
                raise FloatingPointError('Non-finite pre-EA input to global PCTR.')
            total += block.sum(axis=0, dtype=np.float64)
        means.append(total / len(class_rows))
        counts.append(len(class_rows))
    topology, metadata = signal_topology(np.stack(means), np.arange(3, dtype=np.int64), 3)
    if topology is None:
        raise AssertionError('Complete global prototypes unexpectedly produced no topology.')
    metadata.update(counts=counts, num_trials=len(rows), input_shape=[len(rows), *pre.shape[1:]],
                    mean_accumulation='float64 per-class sum in bounded chunks, divided by true class count',
                    statistics_scope='all_source_train_plus_validation')
    return topology, metadata


def augment(x):
    if torch.rand(1, device=x.device) <= .5:
        return x + torch.randn_like(x) * x.std(dim=-1, keepdim=True).clamp_min(1e-6) * .01
    return x


def supervised_loss(net, xb, yb, criterion):
    lam = float(np.random.beta(.4, .4))
    permutation = torch.randperm(len(xb), device=xb.device)
    logits, _ = net(lam * xb + (1 - lam) * xb[permutation])
    return lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yb[permutation])


def make_net(device, variant):
    return Net(use_gasl=variant!="wo_asre", use_dbp_fusion=variant!="wo_dcsf", use_time_branch=variant!="wo_mtre", num_classes=3, input_channels=64, input_time_length=640, fs=160,
               use_dynamic_ea=False, use_feature_align=variant!="wo_alcw", fa_learn_alpha=True,
               attn_gate='none', spec_mode='D').to(device)


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def cpu_model_state(net):
    return {name: tensor.detach().cpu().clone() for name, tensor in net.state_dict().items()}




def finite_output(logits, features):
    if not torch.isfinite(logits).all() or not torch.isfinite(features).all():
        raise FloatingPointError('Non-finite model output.')


@torch.no_grad()
def source_validation(net, aligned, labels, rows, device, batch_size=16):
    net.eval()
    correct, ce_sum = 0, 0.0
    for indices in batches(len(rows), batch_size):
        selected = rows[indices]
        xb = torch.from_numpy(np.asarray(aligned[selected], dtype=np.float32)).to(device)
        yb = torch.from_numpy(np.asarray(labels[selected], dtype=np.int64)).to(device)
        logits, features = net(xb)
        finite_output(logits, features)
        correct += int((logits.argmax(1) == yb).sum())
        ce_sum += float(nn.functional.cross_entropy(logits, yb, reduction='sum'))
    return dict(accuracy=correct / len(rows), ce=ce_sum / len(rows), n_validation=len(rows))


def is_better_validation(candidate, best):
    if best is None:
        return True
    return candidate['accuracy'] > best['accuracy'] or (
        candidate['accuracy'] == best['accuracy'] and candidate['ce'] < best['ce'])


@torch.no_grad()
def evaluate(net, x, y, device, batch_size=16):
    net.eval()
    probabilities = []
    for indices in batches(len(y), batch_size):
        logits, features = net(torch.from_numpy(np.asarray(x[indices], dtype=np.float32)).to(device))
        finite_output(logits, features)
        probabilities.append(logits.softmax(1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    predicted = probabilities.argmax(1)
    metrics = dict(
        accuracy=float(accuracy_score(y, predicted)), kappa=float(cohen_kappa_score(y, predicted)),
        macro_recall=float(recall_score(y, predicted, average='macro', zero_division=0)),
        macro_f1=float(f1_score(y, predicted, average='macro', zero_division=0)),
        macro_auc=float(roc_auc_score(y, probabilities, multi_class='ovr', average='macro')),
        n_test=int(len(y)), confusion_matrix=confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
        true_class_counts=np.bincount(y, minlength=3).tolist(),
        predicted_class_counts=np.bincount(predicted, minlength=3).tolist())
    return metrics, predicted, probabilities


def status(out, **items):
    write_json(out / 'status.json', dict(pid=os.getpid(), updated_at=timestamp(), **items))


def train_stage(net, stage, fold, mode, cfg, folder, aligned, labels, pre, source_rows,
                source_val_rows, x_adapt, global_topology, global_meta):
    """Train without access to target labels or target-test signal arrays."""
    device = torch.device(cfg['device'])
    epochs = cfg['pt_epochs'] if stage == 'pretrain' else cfg['adapt_epochs']
    lr = .001 if stage == 'pretrain' else .0005
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=.1)
    checkpoint = folder / f'{stage}_latest.pt'
    history, start_epoch, best, best_model = [], 0, None, None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['stage'] != stage:
            raise ValueError('Resume stage differs; use a new experiment directory.')
        net.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'])
        start_epoch, history = saved['epoch'] + 1, saved['history']
        best, best_model = saved.get('best'), saved.get('best_model')
        if best_model is not None:
            best_model = {name: tensor.cpu() for name, tensor in best_model.items()}
        del saved
        print(f'RESUME {mode} S{fold:03d} {stage} epoch={start_epoch}', flush=True)
    trace_dir = folder / 'pctr_trace'
    trace_dir.mkdir(exist_ok=True)
    status(folder.parent, state='running', mode=mode, fold=fold, stage=stage,
           epoch=start_epoch, total_epochs=epochs, best_source_validation=best)
    for epoch in range(start_epoch, epochs):
        net.train()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        order_rng = np.random.default_rng(cfg['seed'] + fold * 10000 + epoch + (1000000 if stage == 'adapt' else 0))
        source_batches = batches(len(source_rows), cfg['batch_size'], order_rng)
        target_rng = np.random.default_rng(cfg['seed'] + fold * 100000 + epoch + 2000000)
        unl_batches, unl_cursor = batches(len(x_adapt), cfg['batch_size'], target_rng), 0
        losses, ces, tats, ents = [], [], [], []
        matrices, counts, normalizers, row_trace, target_trace = [], [], [], [], []
        skipped = 0
        for step, indices in enumerate(source_batches):
            if cfg['max_steps'] and step >= cfg['max_steps']:
                break
            row_ids = source_rows[indices]
            batch_labels = np.asarray(labels[row_ids], dtype=np.int64)
            row_trace.append(row_ids)
            topology = None
            if stage == 'pretrain':
                if mode == 'source_global':
                    present = np.bincount(batch_labels, minlength=3)
                    valid = bool(np.all(present > 0))
                    topology = global_topology if valid else None
                    metadata = dict(valid=valid, counts=present.tolist(), normalizer=global_meta['normalizer'])
                else:
                    topology, metadata = signal_topology(pre[row_ids], batch_labels, 3)
                skipped += int(not metadata['valid'])
                matrices.append(np.full((3, 3), np.nan, dtype=np.float32) if topology is None else topology.cpu().numpy())
                counts.append(metadata['counts'])
                normalizers.append(np.nan if metadata['normalizer'] is None else metadata['normalizer'])
            xb = augment(torch.from_numpy(np.asarray(aligned[row_ids], dtype=np.float32)).to(device))
            yb = torch.from_numpy(batch_labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            ce = supervised_loss(net, xb, yb, criterion)
            loss, tat = ce, torch.zeros((), device=device)
            entropy = torch.zeros((), device=device)
            if stage == 'pretrain':
                # The extra feature pass must not advance subsequent paired augmentation RNG.
                state_cpu = torch.get_rng_state()
                state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                _, features = net(xb)
                torch.set_rng_state(state_cpu)
                if state_cuda is not None:
                    torch.cuda.set_rng_state_all(state_cuda)
                tat = prototype_tat_loss(features, yb, topology, 3)
                loss = loss + cfg['pctr_weight'] * tat
            else:
                if unl_cursor == len(unl_batches):
                    unl_batches, unl_cursor = batches(len(x_adapt), cfg['batch_size'], target_rng), 0
                target_indices = unl_batches[unl_cursor]
                target_trace.append(target_indices)
                unl_cursor += 1
                xu = augment(torch.from_numpy(np.asarray(x_adapt[target_indices], dtype=np.float32)).to(device))
                logits_u, features_u = net(xu)
                finite_output(logits_u, features_u)
                probs = logits_u.softmax(1)
                entropy = -(probs * probs.clamp_min(1e-8).log()).sum(1).mean()
                loss = loss + .3 * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss: {stage}, fold={fold}, epoch={epoch}, step={step}')
            loss.backward()
            checks = [torch.isfinite(parameter.grad).all() for parameter in net.parameters() if parameter.grad is not None]
            if not checks or not torch.stack(checks).all():
                raise FloatingPointError(f'Non-finite gradient: {stage}, fold={fold}, epoch={epoch}, step={step}')
            optimizer.step()
            losses.append(float(loss.detach()))
            ces.append(float(ce.detach()))
            tats.append(float(tat.detach()))
            ents.append(float(entropy.detach()))
        if not losses:
            raise RuntimeError('No optimization steps were run.')
        scheduler.step()
        validation = None
        improved = False
        if stage == 'pretrain':
            validation = source_validation(net, aligned, labels, source_val_rows, device, cfg['batch_size'])
            if is_better_validation(validation, best):
                best = dict(epoch=epoch + 1, **validation)
                best_model = cpu_model_state(net)
                improved = True
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        flat_rows = np.concatenate(row_trace)
        stats = dict(
            epoch=epoch + 1, steps=len(losses), source_trials_seen=len(flat_rows), source_trials_available=len(source_rows),
            loss=float(np.mean(losses)), ce=float(np.mean(ces)), pctr=float(np.mean(tats)), entropy=float(np.mean(ents)),
            skipped_pctr=skipped, seconds=elapsed, seconds_per_step=elapsed / len(losses),
            lr=scheduler.get_last_lr()[0],
            source_batch_sizes=[len(rows) for rows in row_trace],
            validation=validation, best_source_validation=best,
            process_rss_mb=psutil.Process().memory_info().rss / 2**20,
            cuda_peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0,
            cuda_peak_reserved_mb=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else 0)
        if target_trace:
            flat_target = np.concatenate(target_trace)
            stats.update(target_exposures=len(flat_target),
                         target_distinct_seen=len(np.unique(flat_target)), target_adapt_available=len(x_adapt))
        history.append(stats)
        if stage == 'pretrain':
            np.savez_compressed(
                trace_dir / f'epoch_{epoch + 1:03d}.npz', matrices=np.asarray(matrices),
                class_counts=np.asarray(counts), normalizers=np.asarray(normalizers), source_rows_flat=flat_rows,
                source_row_offsets=np.cumsum([0] + [len(rows) for rows in row_trace]),
                valid=np.asarray([np.isfinite(matrix).all() for matrix in matrices]))
        # The latest checkpoint embeds best weights, so an interrupted best-file write is recoverable.
        atomic_torch_save(dict(stage=stage, epoch=epoch, model=net.state_dict(),
                               optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
                               history=history, best=best, best_model=best_model), checkpoint)
        if improved:
            atomic_torch_save(dict(selection=best, model=best_model), folder / 'pretrain_best.pt')
        write_json(folder / f'{stage}_history.json', history)
        # Avoid hundreds of batch-size entries in stdout; the complete audit is in the history file.
        print(json.dumps(dict(mode=mode, fold=fold, stage=stage,
                              **{key: value for key, value in stats.items() if key != 'source_batch_sizes'})), flush=True)
        status(folder.parent, state='running', mode=mode, fold=fold, stage=stage,
               epoch=epoch + 1, total_epochs=epochs, stats=stats)
    if stage == 'pretrain':
        if best is None or best_model is None:
            raise RuntimeError('Pretraining finished without a source-validation best checkpoint.')
        net.load_state_dict(best_model, strict=True)
        atomic_torch_save(dict(selection=best, model=best_model), folder / 'pretrain_best.pt')
    return history, best


def write_summary(out, cfg, results):
    if not results:
        return
    keys = ('accuracy', 'kappa', 'macro_recall', 'macro_f1', 'macro_auc')
    means = {key: float(np.mean([result['metrics'][key] for result in results])) for key in keys}
    source_means = {key: float(np.mean([result['source_only_metrics'][key] for result in results])) for key in keys}
    write_json(out / 'summary.json', dict(mode=cfg['mode'], kind=cfg['kind'], completed_folds=len(results),
                                         expected_folds=len(cfg['folds']), seed=cfg['seed'],
                                         results=results, mean=means, source_only_mean=source_means))


def run(cfg):
    if Path(sys.executable).resolve() != INTERPRETER.resolve():
        raise RuntimeError(f'Use the existing dedicated interpreter: {INTERPRETER}')
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_num_interop_threads(1)
    if cfg['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; will not silently fall back.')
    if cfg['kind'] == 'formal' and (cfg['pt_epochs'] != 100 or cfg['adapt_epochs'] != 50 or cfg['max_steps'] != 0):
        raise ValueError('Formal protocol requires 100/50 full epochs.')
    out = ROOT / ('results' if cfg['kind'] == 'formal' else 'validation') / cfg['tag'] / cfg['mode']
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'data' / 'cache'
    if not (cache / 'metadata.json').exists():
        raise FileNotFoundError('Prepare and audit the independent PhysioNet data cache first.')
    previous = out / 'config.json'
    if previous.exists() and json.loads(previous.read_text(encoding='utf-8'))['config'] != cfg:
        raise ValueError('Output configuration differs; choose a new experiment directory.')
    write_json(previous, dict(config=cfg, python=sys.version, executable=sys.executable,
                              torch=torch.__version__, cuda=torch.version.cuda, platform=platform.platform(),
                              pid=os.getpid(), started_at=timestamp()))
    status(out, state='running', mode=cfg['mode'], stage='loading_cache', completed_folds=0,
           expected_folds=len(cfg['folds']))
    print(f'START {cfg["mode"]} pid={os.getpid()} device={cfg["device"]} out={out}', flush=True)
    pre, labels, subjects, trials = load_all(cache)
    aligned = load_aligned(cache)
    if tuple(pre.shape) != (676, 64, 640) or aligned.shape != pre.shape or set(np.unique(subjects)) != set(SUBJECTS):
        raise ValueError('Unexpected PhysioNet cache dimensions or subject population.')
    all_results, begun = [], time.time()
    for fold in cfg['folds']:
        folder = out / f'subject{fold}'
        folder.mkdir(exist_ok=True)
        result_file = folder / 'result.json'
        if result_file.exists():
            result = json.loads(result_file.read_text(encoding='utf-8'))
            all_results.append(result)
            continue
        source, source_train, source_val, adapt, test, manifest = make_split(labels, subjects, trials, fold, 42)
        reference = ROOT.parent/'physionet_mi_pctr_first10'/'results'/'seed42_10fold_native160_maxdist'/'source_global'/f'subject{fold}'/'split.json'
        if manifest != json.loads(reference.read_text(encoding='utf8')):
            raise RuntimeError('Split differs from original seed 42 run')
        write_json(folder / 'split.json', manifest)
        status(out, state='running', mode=cfg['mode'], fold=fold, stage='preparing_fold',
               completed_folds=len(all_results), expected_folds=len(cfg['folds']))
        # Only the small target-adaptation array is materialized; all source EEG stays in the shared mmap.
        x_adapt, whitening = ea_transform(pre[adapt])
        np.save(folder / 'target_adaptation_whitening.npy', whitening)
        global_topology, global_meta = global_signal_topology(pre, labels, source)
        global_meta.update(source_subjects=manifest['source_subjects'], target_subject_excluded=fold,
                           source_training_trials=len(source_train), source_validation_trials_included=len(source_val),
                           total_source_trials=len(source),
                           applied_to_loss=cfg['mode'] == 'source_global',
                           interpretation='Fixed all-source reference, including source-validation labels as '
                                          'explicitly requested by the user; target subject entirely excluded. '
                                          'Batch mode uses the current '
                                          'source-training batch independently, without EMA or historical prototypes.')
        write_json(folder / 'source_global_topology.json', global_meta)
        np.save(folder / 'source_global_topology.npy', global_topology.numpy())
        seed_all(cfg['seed'] + fold * 1000)
        net = make_net(torch.device(cfg['device']), cfg['variant'])
        pre_history, best = train_stage(net, 'pretrain', fold, cfg['mode'], cfg, folder, aligned, labels,
                                        pre, source_train, source_val, x_adapt, global_topology, global_meta)
        source_state = cpu_model_state(net)
        atomic_torch_save(source_state, folder / 'source_model.pth')
        # Source validation labels can now join supervised adaptation after model selection is over.
        adapt_history, _ = train_stage(net, 'adapt', fold, cfg['mode'], cfg, folder, aligned, labels,
                                       pre, source, source_val, x_adapt, global_topology, global_meta)
        atomic_torch_save(net.state_dict(), folder / 'adapted_model.pth')
        status(out, state='running', mode=cfg['mode'], fold=fold, stage='final_evaluation')
        # Target-test signals and labels are first combined only after both training stages are finished.
        x_test, _ = ea_transform(pre[test], reference=whitening)
        test_labels = np.asarray(labels[test], dtype=np.int64)
        metrics, predicted, probabilities = evaluate(net, x_test, test_labels, cfg['device'], cfg['batch_size'])
        np.savez_compressed(folder / 'test_predictions.npz', rows=test, subjects=subjects[test], trials=trials[test],
                            y_true=test_labels, y_pred=predicted, probabilities=probabilities)
        net.load_state_dict(source_state, strict=True)
        source_metrics, source_predicted, source_probabilities = evaluate(net, x_test, test_labels, cfg['device'], cfg['batch_size'])
        np.savez_compressed(folder / 'source_test_predictions.npz', rows=test, subjects=subjects[test], trials=trials[test],
                            y_true=test_labels, y_pred=source_predicted, probabilities=source_probabilities)
        result = dict(
            kind=cfg['kind'], mode=cfg['mode'], fold=fold, seed=cfg['seed'],
            metrics=metrics, source_only_metrics=source_metrics,
            source_trials=len(source), source_train_trials=len(source_train), source_validation_trials=len(source_val),
            target_adapt_trials=len(adapt), target_test_trials=len(test), best_source_validation=best,
            pretrain_epochs=len(pre_history), adapt_epochs=len(adapt_history),
            train_seconds=sum(entry['seconds'] for entry in pre_history + adapt_history),
            pctr_skipped_batches=sum(entry['skipped_pctr'] for entry in pre_history),
            target_test_evaluation_timing='After all pretraining and adaptation; never used for selection.',
            finished_at=timestamp())
        write_json(result_file, result)
        all_results.append(result)
        write_summary(out, cfg, all_results)
        print('FOLD_COMPLETE ' + json.dumps(result), flush=True)
        del net, source_state, x_test, x_adapt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_summary(out, cfg, all_results)
    status(out, state='complete', mode=cfg['mode'], completed_folds=len(all_results),
           expected_folds=len(cfg['folds']), wall_seconds=time.time() - begun)
    print('COMPLETE ' + str(out), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['source_global', 'source_batch'], required=True)
    parser.add_argument('--kind', choices=['smoke', 'benchmark', 'formal'], default='smoke')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--variant', choices=['full','wo_pctr','wo_asre','wo_mtre','wo_dcsf','wo_alcw'], default='full')
    parser.add_argument('--pctr-weight', type=float, choices=[0.03, 0.1], default=0.1,
                        help='PCTR loss weight during source pretraining (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, nargs='+', default=list(SUBJECTS))
    parser.add_argument('--pt-epochs', type=int, default=100)
    parser.add_argument('--adapt-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=2)
    cfg = vars(parser.parse_args())
    if cfg['batch_size'] != 16 or any(fold not in SUBJECTS for fold in cfg['folds']):
        parser.error('This protocol fixes batch_size=16 and the 10 eligible subject IDs.')
    if len(set(cfg['folds'])) != len(cfg['folds']) or min(cfg['pt_epochs'], cfg['adapt_epochs'], cfg['cpu_threads']) < 1 or cfg['max_steps'] < 0:
        parser.error('Folds must be unique, epochs/threads positive, and max_steps nonnegative.')
    if Path(cfg['tag']).name != cfg['tag'] or cfg['tag'] in ('.', '..'):
        parser.error('Tag must be one directory name, not a path.')
    cfg.update(
        dataset='PhysioNet_EEGMMIDB_three_class_10_subjects', input_shape=[64, 640], fs_hz=160,
        epoch_seconds=[0, 4], resampling=False, source_validation_fraction=.2,
        source_selection='best_source_validation_accuracy_then_lower_ce_then_earlier_epoch',
        target_adaptation_fraction='floor(class_count/3)',
        pctr_population='global_all_source_train_plus_validation__batch_current_source_train',
        pctr_target_normalization='exact_maximum_unique_offdiagonal_distance_zero_safe',
        feature_distance_normalization='detached_exact_maximum_unique_offdiagonal_distance_zero_safe',
        target_renormalization_in_loss='none', source_batch_policy='all_rows_merge_singleton_tail',
        target_batch_policy='all_rows_merge_singleton_tail_cycle_shuffled_batches',
        pt_learning_rate=.001, adapt_learning_rate=.0005, weight_decay=.001,
        label_smoothing=.1, mixup_alpha=.4, noise_probability=.5, noise_relative_std=.01,
        split_seed=42, pctr_weight=0. if cfg['variant']=='wo_pctr' else cfg['pctr_weight'], target_entropy_weight=.3)
    try:
        run(cfg)
    except Exception as exc:
        failure = ROOT / ('results' if cfg['kind'] == 'formal' else 'validation') / cfg['tag'] / cfg['mode']
        status(failure, state='failed', mode=cfg['mode'], error=repr(exc))
        raise


if __name__ == '__main__':
    main()
