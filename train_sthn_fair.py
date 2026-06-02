import argparse
import os
import os.path as osp
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import utils as eagle_utils
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


REPO_DIR = osp.dirname(osp.abspath(__file__))
TGB2_DIR = osp.join(REPO_DIR, "TGB2")
TGB2_MODULES_DIR = osp.join(TGB2_DIR, "modules")


def ensure_tgb2_import_path():
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    if TGB2_DIR not in sys.path:
        sys.path.insert(0, TGB2_DIR)
    if TGB2_MODULES_DIR not in sys.path:
        sys.path.insert(0, TGB2_MODULES_DIR)


def import_sthn():
    ensure_tgb2_import_path()
    try:
        from TGB2.modules.sthn import (
            STHN_Interface,
            construct_mini_batch_giant_graph,
            get_mini_batch,
            get_parallel_sampler,
            pre_compute_subgraphs,
            set_seed,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"sampler_core", "torch_sparse", "torchmetrics"}:
            raise ModuleNotFoundError(
                "STHN requires TGB2's compiled sampler and extra dependencies. "
                "Install pybind11/torchmetrics/torch-sparse as needed and run "
                "`cd TGB2/modules && python sthn_sampler_setup.py build_ext --inplace`."
            ) from exc
        raise

    return SimpleNamespace(
        STHN_Interface=STHN_Interface,
        construct_mini_batch_giant_graph=construct_mini_batch_giant_graph,
        get_mini_batch=get_mini_batch,
        get_parallel_sampler=get_parallel_sampler,
        pre_compute_subgraphs=pre_compute_subgraphs,
        set_seed=set_seed,
    )


class STHNNegativeSamplerAdapter:
    """Accept TGB2 STHN's split_mode keyword while using EAGLE's sampler."""

    def __init__(self, sampler):
        self.sampler = sampler

    def query_batch(self, sources, destinations, timestamps, relations, split_mode=None, mode=None):
        split = split_mode if split_mode is not None else mode
        return self.sampler.query_batch(sources, destinations, timestamps, relations, split)


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


def make_dataframe(train_events, val_events, test_events):
    all_events = np.vstack([x for x in (train_events, val_events, test_events) if len(x)])
    df = pd.DataFrame(
        {
            "idx": np.arange(len(all_events), dtype=np.int64),
            "src": all_events[:, 0].astype(np.int64),
            "dst": all_events[:, 2].astype(np.int64),
            "time": all_events[:, 3].astype(np.int64),
            "raw_time": all_events[:, 4].astype(np.int64),
            "label": all_events[:, 1].astype(np.int64),
        }
    )
    n_train = len(train_events)
    n_val = len(val_events)
    n_test = len(test_events)
    train_mask = np.zeros(len(df), dtype=bool)
    val_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    train_mask[:n_train] = True
    val_mask[n_train : n_train + n_val] = True
    test_mask[n_train + n_val : n_train + n_val + n_test] = True
    return df, train_mask, val_mask, test_mask


def build_graph_arrays(df, num_nodes):
    ext_full_indptr = np.zeros(int(num_nodes) + 1, dtype=np.int32)
    ext_full_indices = [[] for _ in range(int(num_nodes))]
    ext_full_ts = [[] for _ in range(int(num_nodes))]
    ext_full_eid = [[] for _ in range(int(num_nodes))]

    for idx, row in df.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])
        if 0 <= src < int(num_nodes):
            ext_full_indices[src].append(dst)
            ext_full_ts[src].append(int(row["time"]))
            ext_full_eid[src].append(int(idx))

    for i in range(int(num_nodes)):
        ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

    ext_full_indices = np.asarray([x for row in ext_full_indices for x in row], dtype=np.int32)
    ext_full_ts = np.asarray([x for row in ext_full_ts for x in row], dtype=np.float32)
    ext_full_eid = np.asarray([x for row in ext_full_eid for x in row], dtype=np.int32)

    for i in range(int(num_nodes)):
        beg = int(ext_full_indptr[i])
        end = int(ext_full_indptr[i + 1])
        if end <= beg:
            continue
        order = np.argsort(ext_full_ts[beg:end], kind="stable")
        ext_full_indices[beg:end] = ext_full_indices[beg:end][order]
        ext_full_ts[beg:end] = ext_full_ts[beg:end][order]
        ext_full_eid[beg:end] = ext_full_eid[beg:end][order]

    return {
        "indptr": ext_full_indptr,
        "indices": ext_full_indices,
        "ts": ext_full_ts,
        "eid": ext_full_eid,
    }


def make_out_dir(args):
    name = (
        f"nsq{args.ns_q}_ns{args.ns_seed}_tpr{args.train_predict_ratio:g}"
        f"_hd{args.hidden_dims}_td{args.time_dims}_ly{args.num_layers}"
        f"_neigh{args.num_neighbors}_edge{args.max_edges}_bs{args.batch_size}"
        f"_lr{args.lr:g}"
    )
    return osp.join("results_sthn_fair", args.dataset, f"seed{args.seed}", name)


def serializable_args(args):
    config = {}
    for key, value in vars(args).items():
        if key in {"train_mask", "val_mask", "test_mask"}:
            config[f"{key}_count"] = int(np.sum(value))
        elif key == "device":
            config[key] = str(value)
        elif isinstance(value, (np.integer,)):
            config[key] = int(value)
        elif isinstance(value, (np.floating,)):
            config[key] = float(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value
    return config


def prepare_sthn_args(args, df, train_mask, val_mask, test_mask, data):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and int(args.gpu) >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.device = device
    args.data = (
        f"fair_{args.dataset}_seed{args.seed}_nsq{args.ns_q}_"
        f"ns{args.ns_seed}_tpr{args.train_predict_ratio:g}"
    )
    args.train_mask = train_mask
    args.val_mask = val_mask
    args.test_mask = test_mask
    args.num_edges = int(len(df))
    args.num_nodes = int(data["num_nodes"])
    args.num_edgeType = int(data["num_rels"])
    args.predict_class = False
    args.use_cached_subgraph = True
    args.use_graph_structure = False
    args.node_feat_dims = 0
    args.edge_feat_dims = int(data["num_rels"]) if args.use_type_feats else 0
    return args


def build_edge_features(args, df):
    if args.use_type_feats:
        labels = torch.from_numpy(df["label"].values.astype(np.int64, copy=False))
        return torch.nn.functional.one_hot(labels, num_classes=int(args.num_edgeType)).float().to(args.device)
    return torch.empty((len(df), 0), dtype=torch.float32, device=args.device)


def build_model(args, sthn):
    edge_predictor_configs = {
        "dim_in_time": int(args.time_dims),
        "dim_in_node": int(args.node_feat_dims),
        "predict_class": 1,
    }
    mixer_configs = {
        "per_graph_size": int(args.max_edges),
        "time_channels": int(args.time_dims),
        "input_channels": int(args.edge_feat_dims),
        "hidden_channels": int(args.hidden_dims),
        "out_channels": int(args.hidden_dims),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "channel_expansion_factor": int(args.channel_expansion_factor),
        "window_size": int(args.window_size),
        "use_single_layer": False,
    }
    model = sthn.STHN_Interface(mixer_configs, edge_predictor_configs)
    return model.to(args.device)


def train_epoch(args, model, optimizer, train_subgraphs, df, edge_feats, sthn):
    model.train()
    cur_df = df[args.train_mask]
    loader_len = len(cur_df.groupby(cur_df.index // int(args.batch_size)))
    cur_inds = 0
    losses = []
    train_time = 0.0
    for ind in range(loader_len):
        inputs, subgraph_node_feats, cur_inds = get_training_inputs(
            args, train_subgraphs, cur_df, edge_feats, cur_inds, ind, sthn
        )
        sync_device(args.device)
        t0 = time.perf_counter()
        loss, _, _ = model(inputs, int(args.neg_samples), subgraph_node_feats)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sync_device(args.device)
        train_time += time.perf_counter() - t0
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0, float(train_time)


def get_training_inputs(args, train_subgraphs, cur_df, edge_feats, cur_inds, ind, sthn):
    from TGB2.modules.sthn import get_inputs_for_ind

    return get_inputs_for_ind(
        train_subgraphs,
        "train",
        int(args.extra_neg_samples),
        int(args.neg_samples),
        None,
        edge_feats,
        cur_df,
        cur_inds,
        ind,
        args,
    )


def scale_edts(edge_dts):
    if len(edge_dts) == 0:
        return torch.empty(0, dtype=torch.float32)
    edge_dts = np.asarray(edge_dts, dtype=np.float32)
    lo = float(np.min(edge_dts))
    hi = float(np.max(edge_dts))
    if hi <= lo:
        return torch.zeros(len(edge_dts), dtype=torch.float32)
    return torch.from_numpy(((edge_dts - lo) / (hi - lo) * 1000.0).astype(np.float32, copy=False))


def make_eval_inputs(args, sampler, rows, negs, edge_feats, sthn):
    batch_size = int(len(rows))
    if batch_size != 1:
        raise ValueError("STHN fair evaluation is intentionally one query per batch.")
    negs = np.asarray(negs, dtype=np.int64)
    root_nodes = np.concatenate(
        [
            rows["src"].values.astype(np.int32),
            rows["dst"].values.astype(np.int32),
            negs.astype(np.int32),
        ]
    )
    root_times = np.tile(rows["time"].values.astype(np.float32), len(negs) + 2)
    subgraphs = sthn.get_mini_batch(sampler, root_nodes, root_times, int(args.sampled_num_hops))
    subgraph_data = sthn.construct_mini_batch_giant_graph(subgraphs, int(args.max_edges))
    eids = subgraph_data["eid"].astype(np.int64, copy=False)
    subgraph_edge_feats = edge_feats[eids].to(args.device)
    subgraph_edts = scale_edts(subgraph_data["edts"]).to(args.device)

    all_inds = []
    all_edge_indptr = subgraph_data["all_edge_indptr"]
    for i in range(len(all_edge_indptr) - 1):
        num_edges = int(all_edge_indptr[i + 1] - all_edge_indptr[i])
        all_inds.extend([(int(args.max_edges) * i + j) for j in range(num_edges)])

    inputs = [
        subgraph_edge_feats,
        subgraph_edts,
        int(len(all_edge_indptr) - 1),
        torch.tensor(all_inds, dtype=torch.long, device=args.device),
    ]
    return inputs, None


@torch.no_grad()
def evaluate_split(args, model, split_name, df, g, edge_feats, neg_sampler, sthn, measure_forward=False):
    model.eval()
    cur_df = df[args.val_mask] if split_name == "val" else df[args.test_mask]
    protocol_sampler = STHNNegativeSamplerAdapter(neg_sampler)
    sampler, _ = sthn.get_parallel_sampler(g, int(args.num_neighbors))
    sampler.reset()

    sums = {}
    forward_time = 0.0
    sample_count = 0
    for _, row in cur_df.iterrows():
        row_df = pd.DataFrame([row])
        neg = protocol_sampler.query_batch(
            row_df["src"].values,
            row_df["dst"].values,
            row_df["raw_time"].values,
            row_df["label"].values,
            split_mode=split_name,
        )[0]
        neg = np.asarray(neg, dtype=np.int64)
        if len(neg) == 0:
            continue

        inputs, node_feats = make_eval_inputs(args, sampler, row_df, neg, edge_feats, sthn)
        sync_device(args.device)
        t0 = time.perf_counter()
        _, pred, _ = model(inputs, len(neg), node_feats)
        sync_device(args.device)
        if measure_forward:
            forward_time += time.perf_counter() - t0

        scores = pred.detach().view(-1).cpu().numpy().astype(np.float32, copy=False)
        pos_scores = scores[:1].reshape(1, 1)
        neg_scores = scores[1:].reshape(1, -1)
        neg_mask = np.ones_like(neg_scores, dtype=bool)
        add_metric_sums(sums, compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask))
        sample_count += 1

    metrics = finalize_metric_sums(sums)
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    return metrics, {"forward_time_s": float(forward_time), "sample_count": int(sample_count)}


def run(args):
    sthn = import_sthn()
    set_random_seed(args.seed)
    sthn.set_seed(args.seed)

    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[STHN-Fair]")
    if int(args.ns_q) <= 0:
        raise ValueError("STHN-Fair requires a fixed positive --ns_q because STHN batches candidates by count.")

    train_events = flatten_snapshots(data["train_list"])
    val_events = flatten_snapshots(data["val_list"])
    test_events = flatten_snapshots(data["test_list"])
    df, train_mask, val_mask, test_mask = make_dataframe(train_events, val_events, test_events)
    args = prepare_sthn_args(args, df, train_mask, val_mask, test_mask, data)

    out_dir = make_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    save_config(out_dir, serializable_args(args))

    g = build_graph_arrays(df, int(data["num_nodes"]))
    edge_feats = build_edge_features(args, df)
    model = build_model(args, sthn)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    print(
        f"[STHN-Fair] model nodes={data['num_nodes']} rels={data['num_rels']} "
        f"device={args.device} edge_feat_dims={args.edge_feat_dims} "
        f"train_batch={args.batch_size} eval_is_per_query=1",
        flush=True,
    )

    reset_cuda_peak(args.device)
    t0 = time.perf_counter()
    train_subgraphs = sthn.pre_compute_subgraphs(args, g, df, mode="train", cache=False)
    train_precompute_time = time.perf_counter() - t0

    checkpoint_path = osp.join(out_dir, "best_model.pt")
    best_val = -float("inf")
    best_epoch = 0
    early_stopped = False
    early_stop_epoch = 0
    train_time_total = float(train_precompute_time)
    epoch_logs = []

    for epoch in range(1, int(args.epochs) + 1):
        loss, train_epoch_time = train_epoch(args, model, optimizer, train_subgraphs, df, edge_feats, sthn)
        train_time_total += train_epoch_time
        log = {
            "epoch": int(epoch),
            "loss": float(loss),
            "train_time_s": float(train_epoch_time),
        }
        do_val = epoch % int(args.evaluate_every) == 0
        if do_val:
            val_metrics, _ = evaluate_split(
                args, model, "val", df, g, edge_feats, data["negative_sampler"], sthn, measure_forward=False
            )
            log["val_mrr_strict"] = float(val_metrics["mrr_strict"])
            log["val_hit@1_strict"] = float(val_metrics["hit@1_strict"])
            log["val_hit@10_strict"] = float(val_metrics["hit@10_strict"])
            if val_metrics["mrr_strict"] > best_val + float(args.tolerance):
                best_val = float(val_metrics["mrr_strict"])
                best_epoch = int(epoch)
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, checkpoint_path)
            log["epochs_since_best"] = int(epoch - best_epoch) if best_epoch else 0
        epoch_logs.append(log)
        print(
            f"[STHN-Fair] epoch={epoch} loss={loss:.5f} train_time={train_epoch_time:.2f}s "
            f"best_val_mrr={max(best_val, 0.0):.5f}",
            flush=True,
        )
        if do_val and best_epoch and int(epoch - best_epoch) >= int(args.patience):
            early_stopped = True
            early_stop_epoch = int(epoch)
            print(
                f"[STHN-Fair] early stop at epoch={epoch}: val_mrr did not improve for "
                f"{epoch - best_epoch} epochs (patience={args.patience}, best_epoch={best_epoch})",
                flush=True,
            )
            break

    train_peak = cuda_peak_allocated(args.device)
    if osp.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        best_epoch = int(args.epochs)
        torch.save({"state_dict": model.state_dict(), "epoch": best_epoch, "val_metrics": {}}, checkpoint_path)

    val_metrics, _ = evaluate_split(
        args, model, "val", df, g, edge_feats, data["negative_sampler"], sthn, measure_forward=False
    )
    reset_cuda_peak(args.device)
    test_metrics, test_profile = evaluate_split(
        args, model, "test", df, g, edge_feats, data["negative_sampler"], sthn, measure_forward=True
    )
    eval_peak = cuda_peak_allocated(args.device)

    metrics = {
        "format": "sthn_fair_v1",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "ns_q": int(args.ns_q),
        "ns_seed": int(args.ns_seed),
        "train_predict_ratio": float(args.train_predict_ratio),
        "best_epoch": int(best_epoch),
        "best_val_mrr": float(best_val),
        "early_stop_metric": "val_mrr_strict",
        "early_stopped": bool(early_stopped),
        "early_stop_epoch": int(early_stop_epoch),
        "patience": int(args.patience),
        "train_precompute_time_s": float(train_precompute_time),
        "train_time_s": float(train_time_total),
        "train_peak_allocated_bytes": train_peak,
        "eval_peak_allocated_bytes": eval_peak,
        "test_forward_time_s": float(test_profile["forward_time_s"]),
        "test_inference_sample_count": int(test_profile["sample_count"]),
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
            "TGB2 STHN subgraph sampler, Patch_Encoding, and STHN_Interface are reused. "
            "Training keeps STHN's internal random negative sampling. Final val/test "
            "score exactly one positive plus EAGLE protocol negatives and compute strict metrics."
        ),
    }
    save_metrics(out_dir, metrics)
    print(
        f"[STHN-Fair] final val_mrr={metrics['val_mrr']:.6f} test_mrr={metrics['test_mrr']:.6f} "
        f"test_hit1={metrics['test_hit1']:.6f} test_hit10={metrics['test_hit10']:.6f}",
        flush=True,
    )
    print(
        f"[STHN-Fair] train_time={train_time_total:.3f}s "
        f"test_forward_time={test_profile['forward_time_s']:.3f}s "
        f"test_samples={test_profile['sample_count']} "
        f"train_peak={format_bytes(train_peak)} eval_peak={format_bytes(eval_peak)} saved -> {out_dir}",
        flush=True,
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser("Fair STHN trainer for EAGLE TKG/THG protocols.")
    parser.add_argument("--dataset", type=str, required=True, choices=eagle_utils.SUPPORTED_DATASETS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--max_edges", type=int, default=50)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--neg_samples", type=int, default=1)
    parser.add_argument("--extra_neg_samples", type=int, default=5)
    parser.add_argument("--num_neighbors", type=int, default=50)
    parser.add_argument("--channel_expansion_factor", type=int, default=2)
    parser.add_argument("--sampled_num_hops", type=int, default=1)
    parser.add_argument("--time_dims", type=int, default=100)
    parser.add_argument("--hidden_dims", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--use_type_feats", action="store_true", default=True)
    parser.add_argument("--no_type_feats", dest="use_type_feats", action="store_false")

    args = parser.parse_args()
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch_size must be positive")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if int(args.evaluate_every) <= 0:
        raise ValueError("--evaluate-every must be positive")
    if int(args.patience) <= 0:
        raise ValueError("--patience must be positive")
    if int(args.max_edges) <= 0 or int(args.window_size) <= 0:
        raise ValueError("--max_edges and --window_size must be positive")
    if int(args.max_edges) % int(args.window_size) != 0:
        raise ValueError("--max_edges must be divisible by --window_size for STHN Patch_Encoding")
    if int(args.neg_samples) <= 0 or int(args.extra_neg_samples) <= 0:
        raise ValueError("--neg_samples and --extra_neg_samples must be positive")
    if int(args.num_neighbors) <= 0 or int(args.sampled_num_hops) <= 0:
        raise ValueError("--num_neighbors and --sampled_num_hops must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
