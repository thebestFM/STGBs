import argparse
import os
import os.path as osp
import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from utils import (
    add_metric_sums,
    compute_ranking_metric_sums,
    describe_loaded_data,
    finalize_metric_sums,
    load_datasets,
    save_config,
    save_metrics,
    set_random_seed,
)
import utils as eagle_utils


REPO_DIR = osp.dirname(osp.abspath(__file__))
TGB1_DIR = osp.join(REPO_DIR, "TGB1")


def import_tgb1():
    """Import TGB1/DyGLib modules without colliding with this repo's top-level utils.py."""
    if not osp.isdir(TGB1_DIR):
        raise FileNotFoundError(f"TGB1 directory not found: {TGB1_DIR}")

    saved_path = list(sys.path)
    saved_modules = {
        name: sys.modules.get(name)
        for name in (
            "utils",
            "utils.utils",
            "models",
            "models.modules",
            "models.MemoryModel",
            "models.GraphMixer",
        )
    }
    missing = {name for name, value in saved_modules.items() if value is None}
    try:
        sys.path.insert(0, TGB1_DIR)
        for name in saved_modules:
            sys.modules.pop(name, None)

        # TGB1 modules use absolute imports such as `from utils.utils import ...`
        # and `from models.modules import ...`.  The current repo also has a
        # top-level utils.py, so expose TGB1/utils and TGB1/models as temporary
        # packages while importing the official modules.
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [osp.join(TGB1_DIR, "utils")]
        models_pkg = types.ModuleType("models")
        models_pkg.__path__ = [osp.join(TGB1_DIR, "models")]
        sys.modules["utils"] = utils_pkg
        sys.modules["models"] = models_pkg

        from models.MemoryModel import MemoryModel
        from models.GraphMixer import GraphMixer
        from utils.utils import NeighborSampler
    finally:
        sys.path[:] = saved_path
        for name in saved_modules:
            if name in missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved_modules[name]

    return SimpleNamespace(
        MemoryModel=MemoryModel,
        GraphMixer=GraphMixer,
        NeighborSampler=NeighborSampler,
    )


class RelationAwareLinkPredictor(nn.Module):
    """A light relation-aware scorer on top of TGB1 src/dst temporal embeddings."""

    def __init__(self, node_dim, num_rels, rel_dim=None, hidden_dim=None, dropout=0.1):
        super().__init__()
        rel_dim = int(node_dim if rel_dim is None else rel_dim)
        hidden_dim = int(max(node_dim, rel_dim) if hidden_dim is None else hidden_dim)
        self.rel_emb = nn.Embedding(int(num_rels), rel_dim)
        in_dim = int(node_dim) * 4 + rel_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src_emb, dst_emb, rel_ids):
        rel = self.rel_emb(rel_ids.long())
        feat = torch.cat([src_emb, dst_emb, src_emb * dst_emb, torch.abs(src_emb - dst_emb), rel], dim=1)
        return self.net(feat).squeeze(-1)


class TGB1RelationModel(nn.Module):
    def __init__(self, backbone, predictor, model_kind):
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor
        self.model_kind = model_kind

    @property
    def is_memory_model(self):
        return self.model_kind == "tgn"

    def set_neighbor_sampler(self, sampler):
        self.backbone.set_neighbor_sampler(sampler)

    def reset_memory(self):
        if self.is_memory_model:
            self.backbone.memory_bank.__init_memory_bank__()

    def backup_memory(self):
        if not self.is_memory_model:
            return None
        return self.backbone.memory_bank.backup_memory_bank()

    def reload_memory(self, backup):
        if self.is_memory_model and backup is not None:
            self.backbone.memory_bank.reload_memory_bank(backup)

    def detach_memory(self):
        if self.is_memory_model:
            self.backbone.memory_bank.detach_memory_bank()


def reset_cuda_peak(device):
    if getattr(device, "type", None) != "cuda":
        return False
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return True


def cuda_peak_allocated(device):
    if getattr(device, "type", None) != "cuda":
        return None
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def sync_device(device):
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def format_bytes(value):
    if value is None:
        return "n/a"
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}GiB"


def flatten_snapshots(snapshot_list):
    chunks = []
    for events, t_norm, t_orig in snapshot_list:
        if len(events) == 0:
            continue
        t_norm_col = np.full((len(events), 1), int(t_norm), dtype=np.int64)
        t_orig_col = np.full((len(events), 1), int(t_orig), dtype=np.int64)
        chunks.append(np.hstack((events.astype(np.int64, copy=False), t_norm_col, t_orig_col)))
    if not chunks:
        return np.empty((0, 5), dtype=np.int64)
    return np.vstack(chunks).astype(np.int64, copy=False)


def make_events(data):
    train = flatten_snapshots(data["train_list"])
    val = flatten_snapshots(data["val_list"])
    test = flatten_snapshots(data["test_list"])
    all_events = np.vstack([x for x in (train, val, test) if len(x)])
    edge_ids = np.arange(1, len(all_events) + 1, dtype=np.int64)

    train_len = len(train)
    val_len = len(val)
    splits = {
        "train": np.arange(0, train_len, dtype=np.int64),
        "val": np.arange(train_len, train_len + val_len, dtype=np.int64),
        "test": np.arange(train_len + val_len, len(all_events), dtype=np.int64),
    }
    return all_events, edge_ids, splits


def build_edge_features(events, num_rels):
    labels = events[:, 1].astype(np.int64, copy=False)
    feats = np.zeros((len(events) + 1, int(num_rels)), dtype=np.float32)
    valid = (labels >= 0) & (labels < int(num_rels))
    feats[np.arange(1, len(events) + 1, dtype=np.int64)[valid], labels[valid]] = 1.0
    return feats


def build_neighbor_sampler(tgb1, events, edge_ids, indices, num_nodes, strategy, time_scaling_factor, seed):
    adj_list = [[] for _ in range(int(num_nodes) + 1)]
    for idx in indices:
        row = events[int(idx)]
        src = int(row[0]) + 1
        rel = int(row[1])
        dst = int(row[2]) + 1
        t_norm = float(row[3])
        eid = int(edge_ids[int(idx)])
        if 1 <= src <= int(num_nodes) and 1 <= dst <= int(num_nodes):
            adj_list[src].append((dst, eid, t_norm))
            adj_list[dst].append((src, eid, t_norm))
    return tgb1.NeighborSampler(
        adj_list=adj_list,
        sample_neighbor_strategy=strategy,
        time_scaling_factor=float(time_scaling_factor),
        seed=seed,
    )


def destination_pool(data):
    sampler = data["negative_sampler"]
    first = int(getattr(sampler, "first_dst_id", 0))
    last = int(getattr(sampler, "last_dst_id", int(data["num_nodes"]) - 1))
    return np.arange(first + 1, last + 2, dtype=np.int64)


def sample_training_negatives(rng, dst_ids_shifted, pool_shifted):
    if len(pool_shifted) == 0:
        return dst_ids_shifted.copy()
    neg = rng.choice(pool_shifted, size=len(dst_ids_shifted), replace=True).astype(np.int64, copy=False)
    if len(pool_shifted) > 1:
        same = neg == dst_ids_shifted
        while np.any(same):
            neg[same] = rng.choice(pool_shifted, size=int(np.sum(same)), replace=True)
            same = neg == dst_ids_shifted
    return neg


def encode_edges(model, src, dst, times, edge_ids=None, positive=False, args=None):
    if model.model_kind == "tgn":
        return model.backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src,
            dst_node_ids=dst,
            node_interact_times=times,
            edge_ids=edge_ids if positive else None,
            edges_are_positive=bool(positive),
            num_neighbors=int(args.num_neighbors),
        )
    return model.backbone.compute_src_dst_node_temporal_embeddings(
        src_node_ids=src,
        dst_node_ids=dst,
        node_interact_times=times,
        num_neighbors=int(args.num_neighbors),
        time_gap=int(args.time_gap),
    )


def make_model(args, data, events, tgb1):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and int(args.gpu) >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.device = device

    num_nodes = int(data["num_nodes"])
    num_rels = int(data["num_rels"])
    node_raw_features = np.zeros((num_nodes + 1, int(args.node_feat_dim)), dtype=np.float32)
    edge_raw_features = build_edge_features(events, num_rels)
    empty_sampler = tgb1.NeighborSampler(
        adj_list=[[] for _ in range(num_nodes + 1)],
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=float(args.time_scaling_factor),
        seed=args.seed,
    )

    if args.model_kind == "tgn":
        backbone = tgb1.MemoryModel(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=empty_sampler,
            time_feat_dim=int(args.time_feat_dim),
            model_name="TGN",
            num_layers=int(args.num_layers),
            num_heads=int(args.num_heads),
            dropout=float(args.dropout),
            device=str(device),
        )
    elif args.model_kind == "graphmixer":
        backbone = tgb1.GraphMixer(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=empty_sampler,
            time_feat_dim=int(args.time_feat_dim),
            num_tokens=int(args.num_neighbors),
            num_layers=int(args.num_layers),
            token_dim_expansion_factor=float(args.token_dim_expansion_factor),
            channel_dim_expansion_factor=float(args.channel_dim_expansion_factor),
            dropout=float(args.dropout),
            device=str(device),
        )
    else:
        raise ValueError(f"unsupported model_kind: {args.model_kind}")

    predictor = RelationAwareLinkPredictor(
        node_dim=int(args.node_feat_dim),
        num_rels=num_rels,
        rel_dim=int(args.rel_dim),
        hidden_dim=int(args.predictor_hidden_dim),
        dropout=float(args.dropout),
    )
    return TGB1RelationModel(backbone, predictor, args.model_kind).to(device), device


def checkpoint_payload(model, epoch, val_metrics=None, train_loss=None):
    payload = {
        "state_dict": model.state_dict(),
        "epoch": int(epoch),
        "val_metrics": val_metrics or {},
        "train_loss": None if train_loss is None else float(train_loss),
    }
    if model.is_memory_model:
        payload["memory_backup"] = model.backup_memory()
    return payload


def load_checkpoint(model, path, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if model.is_memory_model:
        model.reload_memory(ckpt.get("memory_backup"))
    return ckpt


def train_one_epoch(args, model, optimizer, events, edge_ids, train_indices, train_sampler, dst_pool, rng):
    model.train()
    model.set_neighbor_sampler(train_sampler)
    if model.is_memory_model:
        model.reset_memory()

    losses = []
    start_time = time.perf_counter()
    loss_fn = nn.BCEWithLogitsLoss()
    batch_size = int(args.batch_size)
    for start in range(0, len(train_indices), batch_size):
        idx = train_indices[start : start + batch_size]
        if len(idx) == 0:
            continue
        batch = events[idx]
        src = batch[:, 0].astype(np.int64, copy=False) + 1
        rel = batch[:, 1].astype(np.int64, copy=False)
        dst = batch[:, 2].astype(np.int64, copy=False) + 1
        times = batch[:, 3].astype(np.float32, copy=False)
        eids = edge_ids[idx].astype(np.int64, copy=False)
        neg_dst = sample_training_negatives(rng, dst, dst_pool)

        if model.is_memory_model:
            neg_src_emb, neg_dst_emb = encode_edges(model, src, neg_dst, times, positive=False, args=args)
            pos_src_emb, pos_dst_emb = encode_edges(model, src, dst, times, edge_ids=eids, positive=True, args=args)
        else:
            pos_src_emb, pos_dst_emb = encode_edges(model, src, dst, times, positive=False, args=args)
            neg_src_emb, neg_dst_emb = encode_edges(model, src, neg_dst, times, positive=False, args=args)

        rel_t = torch.from_numpy(rel).long().to(args.device)
        pos_logits = model.predictor(pos_src_emb, pos_dst_emb, rel_t)
        neg_logits = model.predictor(neg_src_emb, neg_dst_emb, rel_t)
        logits = torch.cat((pos_logits, neg_logits), dim=0)
        labels = torch.cat((torch.ones_like(pos_logits), torch.zeros_like(neg_logits)), dim=0)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_norm))
        optimizer.step()
        model.detach_memory()
        losses.append(float(loss.detach().cpu().item()))

    sync_device(args.device)
    elapsed = time.perf_counter() - start_time
    return float(np.mean(losses)) if losses else 0.0, float(elapsed)


def score_negative_candidates(args, model, src, rel, times, neg_arr, neg_mask):
    batch_size, width = neg_arr.shape
    neg_scores = np.zeros((batch_size, width), dtype=np.float32)
    valid_rows, valid_cols = np.nonzero(neg_mask)
    if len(valid_rows) == 0:
        return neg_scores

    chunk_size = max(1, int(args.eval_candidate_batch_size))
    for start in range(0, len(valid_rows), chunk_size):
        end = min(start + chunk_size, len(valid_rows))
        rows = valid_rows[start:end]
        cols = valid_cols[start:end]
        flat_src = src[rows]
        flat_dst = neg_arr[rows, cols].astype(np.int64, copy=False) + 1
        flat_rel = rel[rows]
        flat_times = times[rows]
        src_emb, dst_emb = encode_edges(model, flat_src, flat_dst, flat_times, positive=False, args=args)
        rel_t = torch.from_numpy(flat_rel.astype(np.int64, copy=False)).long().to(args.device)
        logits = model.predictor(src_emb, dst_emb, rel_t).detach().cpu().numpy().astype(np.float32, copy=False)
        neg_scores[rows, cols] = logits
    return neg_scores


@torch.no_grad()
def evaluate_split(args, model, split_name, events, edge_ids, split_indices, full_sampler, neg_sampler, measure_forward=False):
    model.eval()
    model.set_neighbor_sampler(full_sampler)

    sums = {}
    forward_time = 0.0
    sample_count = 0
    batch_size = int(args.eval_batch_size)
    ordered = split_indices

    for start in range(0, len(ordered), batch_size):
        idx = ordered[start : start + batch_size]
        if len(idx) == 0:
            continue
        batch = events[idx]
        src_orig = batch[:, 0].astype(np.int64, copy=False)
        rel = batch[:, 1].astype(np.int64, copy=False)
        dst_orig = batch[:, 2].astype(np.int64, copy=False)
        times = batch[:, 3].astype(np.float32, copy=False)
        raw_times = batch[:, 4].astype(np.int64, copy=False)

        negs = neg_sampler.query_batch(src_orig, dst_orig, raw_times, rel, split_name)
        width = max((len(x) for x in negs), default=0) or 1
        neg_arr = np.full((len(batch), width), -1, dtype=np.int64)
        for row_id, row in enumerate(negs):
            if len(row):
                neg_arr[row_id, : len(row)] = np.asarray(row, dtype=np.int64)
        neg_mask = neg_arr != -1

        src = src_orig + 1
        dst = dst_orig + 1
        eids = edge_ids[idx].astype(np.int64, copy=False)

        sync_device(args.device)
        t0 = time.perf_counter()
        neg_scores = score_negative_candidates(args, model, src, rel, times, neg_arr, neg_mask)
        if model.is_memory_model:
            pos_src_emb, pos_dst_emb = encode_edges(model, src, dst, times, edge_ids=eids, positive=True, args=args)
        else:
            pos_src_emb, pos_dst_emb = encode_edges(model, src, dst, times, positive=False, args=args)
        rel_t = torch.from_numpy(rel).long().to(args.device)
        pos_scores = model.predictor(pos_src_emb, pos_dst_emb, rel_t).detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
        sync_device(args.device)
        if measure_forward:
            forward_time += time.perf_counter() - t0
        if model.is_memory_model:
            model.detach_memory()

        add_metric_sums(sums, compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask))
        sample_count += int(len(batch))

    metrics = finalize_metric_sums(sums)
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    return metrics, {"forward_time_s": float(forward_time), "sample_count": int(sample_count)}


def make_out_dir(args):
    prefix = "results_tgn_fair" if args.model_kind == "tgn" else "results_graphmixer_fair"
    name = (
        f"nsq{args.ns_q}_ns{args.ns_seed}_tpr{args.train_predict_ratio:g}"
        f"_d{args.node_feat_dim}_td{args.time_feat_dim}_rd{args.rel_dim}"
        f"_ly{args.num_layers}_neigh{args.num_neighbors}_bs{args.batch_size}_lr{args.lr:g}"
    )
    return osp.join(prefix, args.dataset, f"seed{args.seed}", name)


def serializable_args(args):
    out = {}
    for key, value in vars(args).items():
        if key == "device":
            out[key] = str(value)
        elif isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def run_fair(args):
    tgb1 = import_tgb1()
    set_random_seed(args.seed)

    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    prefix = "[TGN-Fair]" if args.model_kind == "tgn" else "[GraphMixer-Fair]"
    describe_loaded_data(data, prefix=prefix)
    if int(args.ns_q) <= 0:
        raise ValueError("TGB1 fair TGN/GraphMixer require a fixed positive --ns_q")

    events, edge_ids, splits = make_events(data)
    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]

    out_dir = make_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    model, device = make_model(args, data, events, tgb1)
    save_config(out_dir, serializable_args(args))

    train_sampler = build_neighbor_sampler(
        tgb1,
        events,
        edge_ids,
        train_indices,
        int(data["num_nodes"]),
        args.sample_neighbor_strategy,
        args.time_scaling_factor,
        args.seed,
    )
    full_sampler = build_neighbor_sampler(
        tgb1,
        events,
        edge_ids,
        np.arange(len(events), dtype=np.int64),
        int(data["num_nodes"]),
        args.sample_neighbor_strategy,
        args.time_scaling_factor,
        args.seed,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.RandomState(int(args.seed) + 97)
    dst_pool = destination_pool(data)

    print(
        f"{prefix} model nodes={data['num_nodes']} rels={data['num_rels']} device={device} "
        f"node_dim={args.node_feat_dim} time_dim={args.time_feat_dim} rel_dim={args.rel_dim} "
        f"batch={args.batch_size} eval_batch={args.eval_batch_size} candidates/chunk={args.eval_candidate_batch_size}",
        flush=True,
    )

    best_val = -float("inf")
    best_val_epoch = 0
    best_train_loss = float("inf")
    best_train_loss_epoch = 0
    early_stopped = False
    early_stop_epoch = 0
    train_time_total = 0.0
    epoch_logs = []
    val_checkpoint_path = osp.join(out_dir, "best_val_model.pt")
    loss_checkpoint_path = osp.join(out_dir, "best_train_loss_model.pt")

    reset_cuda_peak(device)
    for epoch in range(1, int(args.epochs) + 1):
        loss, train_time = train_one_epoch(args, model, optimizer, events, edge_ids, train_indices, train_sampler, dst_pool, rng)
        train_time_total += train_time
        if loss < best_train_loss - float(args.tolerance):
            best_train_loss = float(loss)
            best_train_loss_epoch = int(epoch)
            torch.save(checkpoint_payload(model, epoch, train_loss=loss), loss_checkpoint_path)

        log = {
            "epoch": int(epoch),
            "loss": float(loss),
            "train_time_s": float(train_time),
            "best_train_loss": float(best_train_loss),
        }
        do_val = int(args.evaluate_every) > 0 and epoch % int(args.evaluate_every) == 0
        if do_val:
            memory_backup = model.backup_memory()
            val_metrics, _ = evaluate_split(
                args,
                model,
                "val",
                events,
                edge_ids,
                val_indices,
                full_sampler,
                data["negative_sampler"],
                measure_forward=False,
            )
            model.reload_memory(memory_backup)
            log["val_mrr_strict"] = float(val_metrics["mrr_strict"])
            log["val_hit@1_strict"] = float(val_metrics["hit@1_strict"])
            log["val_hit@10_strict"] = float(val_metrics["hit@10_strict"])
            if val_metrics["mrr_strict"] > best_val + float(args.tolerance):
                best_val = float(val_metrics["mrr_strict"])
                best_val_epoch = int(epoch)
                torch.save(checkpoint_payload(model, epoch, val_metrics=val_metrics, train_loss=loss), val_checkpoint_path)
            log["epochs_since_best_val"] = int(epoch - best_val_epoch) if best_val_epoch else 0

        epoch_logs.append(log)
        print(
            f"{prefix} epoch={epoch} loss={loss:.5f} train_time={train_time:.2f}s "
            f"best_train_loss={best_train_loss:.5f}@{best_train_loss_epoch} "
            f"best_val_mrr={max(best_val, 0.0):.5f}",
            flush=True,
        )
        if best_train_loss_epoch and int(epoch - best_train_loss_epoch) >= int(args.patience):
            early_stopped = True
            early_stop_epoch = int(epoch)
            print(
                f"{prefix} early stop at epoch={epoch}: train loss did not improve for "
                f"{epoch - best_train_loss_epoch} epochs (patience={args.patience})",
                flush=True,
            )
            break

    train_peak = cuda_peak_allocated(device)
    selected_by = "val_mrr" if best_val_epoch and osp.exists(val_checkpoint_path) else "train_loss"
    selected_path = val_checkpoint_path if selected_by == "val_mrr" else loss_checkpoint_path
    if osp.exists(selected_path):
        ckpt = load_checkpoint(model, selected_path, device)
        best_epoch = int(ckpt.get("epoch", best_val_epoch or best_train_loss_epoch))
    else:
        best_epoch = int(args.epochs)
        selected_by = "final"
        torch.save(checkpoint_payload(model, best_epoch), loss_checkpoint_path)

    val_metrics, val_profile = evaluate_split(
        args,
        model,
        "val",
        events,
        edge_ids,
        val_indices,
        full_sampler,
        data["negative_sampler"],
        measure_forward=False,
    )
    reset_cuda_peak(device)
    test_metrics, test_profile = evaluate_split(
        args,
        model,
        "test",
        events,
        edge_ids,
        test_indices,
        full_sampler,
        data["negative_sampler"],
        measure_forward=True,
    )
    eval_peak = cuda_peak_allocated(device)
    reported_best_val = float(best_val) if np.isfinite(best_val) else float(val_metrics["mrr_strict"])

    metrics = {
        "format": f"{args.model_kind}_fair_v1",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "ns_q": int(args.ns_q),
        "ns_seed": int(args.ns_seed),
        "train_predict_ratio": float(args.train_predict_ratio),
        "best_epoch": int(best_epoch),
        "best_val_epoch": int(best_val_epoch),
        "best_train_loss_epoch": int(best_train_loss_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_mrr": reported_best_val,
        "selected_checkpoint_by": selected_by,
        "early_stopped": bool(early_stopped),
        "early_stop_epoch": int(early_stop_epoch),
        "patience": int(args.patience),
        "train_time_s": float(train_time_total),
        "train_peak_allocated_bytes": train_peak,
        "eval_peak_allocated_bytes": eval_peak,
        "test_forward_time_s": float(test_profile["forward_time_s"]),
        "test_inference_sample_count": int(test_profile["sample_count"]),
        "val_profile": val_profile,
        "test_profile": test_profile,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_mrr": float(val_metrics["mrr_strict"]),
        "val_hit1": float(val_metrics["hit@1_strict"]),
        "val_hit10": float(val_metrics["hit@10_strict"]),
        "test_mrr": float(test_metrics["mrr_strict"]),
        "test_hit1": float(test_metrics["hit@1_strict"]),
        "test_hit10": float(test_metrics["hit@10_strict"]),
        "epoch_logs": epoch_logs,
        "model_note": (
            "TGB1 backbone is reused with relation one-hot historical edge features and a relation-aware "
            "link predictor. Training uses chronological random destination negatives from the same "
            "destination pool as the protocol. Final val/test score exactly one positive plus EAGLE "
            "protocol negatives and compute strict metrics. TGN memory is updated online with positive "
            "edges; GraphMixer is stateless and uses temporal neighbor sampling."
        ),
    }
    save_metrics(out_dir, metrics)
    print(
        f"{prefix} final val_mrr={metrics['val_mrr']:.6f} test_mrr={metrics['test_mrr']:.6f} "
        f"test_hit1={metrics['test_hit1']:.6f} test_hit10={metrics['test_hit10']:.6f} "
        f"selected={selected_by}@epoch{best_epoch}",
        flush=True,
    )
    print(
        f"{prefix} train_time={train_time_total:.3f}s test_forward_time={test_profile['forward_time_s']:.3f}s "
        f"test_samples={test_profile['sample_count']} train_peak={format_bytes(train_peak)} "
        f"eval_peak={format_bytes(eval_peak)} saved -> {out_dir}",
        flush=True,
    )
    return metrics


def add_common_args(parser, model_kind):
    parser.add_argument("--model_kind", type=str, default=model_kind, choices=("tgn", "graphmixer"))
    parser.add_argument("--dataset", type=str, required=True, choices=eagle_utils.SUPPORTED_DATASETS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_candidate_batch_size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_norm", type=float, default=1.0)

    parser.add_argument("--node_feat_dim", type=int, default=64)
    parser.add_argument("--time_feat_dim", type=int, default=64)
    parser.add_argument("--rel_dim", type=int, default=64)
    parser.add_argument("--predictor_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=1 if model_kind == "tgn" else 2)
    parser.add_argument("--num_neighbors", type=int, default=10 if model_kind == "tgn" else 20)
    parser.add_argument("--sample_neighbor_strategy", type=str, default="recent", choices=("recent", "uniform", "time_interval_aware"))
    parser.add_argument("--time_scaling_factor", type=float, default=0.0)

    if model_kind == "tgn":
        parser.add_argument("--num_heads", type=int, default=2)
    else:
        parser.add_argument("--time_gap", type=int, default=200)
        parser.add_argument("--token_dim_expansion_factor", type=float, default=0.5)
        parser.add_argument("--channel_dim_expansion_factor", type=float, default=2.0)


def validate_common_args(args):
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if int(args.batch_size) <= 0 or int(args.eval_batch_size) <= 0:
        raise ValueError("--batch_size and --eval_batch_size must be positive")
    if int(args.eval_candidate_batch_size) <= 0:
        raise ValueError("--eval_candidate_batch_size must be positive")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if int(args.evaluate_every) == 0 or int(args.evaluate_every) < -1:
        raise ValueError("--evaluate-every must be -1 or a positive integer")
    if int(args.patience) <= 0:
        raise ValueError("--patience must be positive")
    if int(args.node_feat_dim) <= 0 or int(args.time_feat_dim) <= 0 or int(args.rel_dim) <= 0:
        raise ValueError("--node_feat_dim, --time_feat_dim, and --rel_dim must be positive")
    if int(args.num_neighbors) <= 0:
        raise ValueError("--num_neighbors must be positive")
    if args.model_kind == "tgn" and (int(args.node_feat_dim) + int(args.time_feat_dim)) % int(args.num_heads) != 0:
        raise ValueError("For TGN attention, node_feat_dim + time_feat_dim must be divisible by --num_heads")
    if args.model_kind == "graphmixer" and int(args.time_gap) <= 0:
        raise ValueError("--time_gap must be positive")
    return args


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model_kind", type=str, default="tgn", choices=("tgn", "graphmixer"))
    pre_args, _ = pre_parser.parse_known_args()
    model_kind = pre_args.model_kind

    parser = argparse.ArgumentParser("Fair TGB1 TGN-r/GraphMixer-r trainer for EAGLE TKG/THG protocols.")
    add_common_args(parser, model_kind)
    args = parser.parse_args()
    return validate_common_args(args)


def parse_common_args(model_kind, description):
    parser = argparse.ArgumentParser(description)
    add_common_args(parser, model_kind)
    args = parser.parse_args()
    args.model_kind = model_kind
    return validate_common_args(args)


if __name__ == "__main__":
    run_fair(parse_args())
