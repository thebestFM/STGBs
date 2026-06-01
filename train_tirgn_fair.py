import argparse
import json
import os
import os.path as osp
import random
import sys
import time
from collections import defaultdict

import numpy as np

import utils as eagle_utils
from utils import (
    add_metric_sums,
    collect_eval_batch,
    compute_ranking_metric_sums,
    describe_loaded_data,
    finalize_metric_sums,
    load_datasets,
    save_config,
    save_metrics,
    set_random_seed,
)


REPO_DIR = osp.dirname(osp.abspath(__file__))
TIRGN_DIR = osp.join(REPO_DIR, "baseline_TiRGN")


def ensure_tirgn_import_path():
    """Resolve TiRGN's own rgcn/ and src/ packages from baseline_TiRGN."""
    if not osp.isdir(osp.join(TIRGN_DIR, "rgcn")) or not osp.isdir(osp.join(TIRGN_DIR, "src")):
        raise FileNotFoundError(f"TiRGN rgcn/src directories not found under {TIRGN_DIR}")
    if TIRGN_DIR in sys.path:
        sys.path.remove(TIRGN_DIR)
    sys.path.insert(0, TIRGN_DIR)


def import_tirgn():
    ensure_tirgn_import_path()
    import torch
    from rgcn.knowledge_graph import _read_triplets_as_list
    from rgcn.utils import build_sub_graph
    from src.rrgcn import RecurrentRGCN

    return torch, RecurrentRGCN, build_sub_graph, _read_triplets_as_list


def reset_cuda_peak(torch, device):
    if getattr(device, "type", None) != "cuda":
        return False
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return True


def cuda_peak_allocated(torch, device):
    if getattr(device, "type", None) != "cuda":
        return None
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def sync_device(torch, device):
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


def snapshot_to_quad(snapshot):
    events, t_norm, _ = snapshot
    t_col = np.full((len(events), 1), int(t_norm), dtype=np.int64)
    return np.hstack((events.astype(np.int64, copy=False), t_col))


def make_quad_snapshots(snapshot_list):
    return [snapshot_to_quad(s) for s in snapshot_list]


def snapshot_raw_times(snapshot_list):
    return [int(t_orig) for _, _, t_orig in snapshot_list]


def infer_time_interval(quad_snapshots):
    times = [int(s[0, 3]) for s in quad_snapshots if len(s)]
    if len(times) < 2:
        return 1
    diffs = np.diff(np.asarray(sorted(set(times)), dtype=np.int64))
    diffs = diffs[diffs > 0]
    return int(diffs.min()) if len(diffs) else 1


def infer_num_times(quad_snapshots, time_interval):
    times = [int(s[0, 3]) for s in quad_snapshots if len(s)]
    if not times:
        return 1
    interval = max(int(time_interval), 1)
    return int(max(times) // interval) + 1


class TemporalHistoryIndex:
    """Sparse local-global history lookup over the protocol-provided edges."""

    def __init__(self, quad_snapshots, num_nodes, num_rels):
        self.num_nodes = int(num_nodes)
        self.num_rels = int(num_rels)
        self.tail = defaultdict(list)
        self.rel = defaultdict(list)
        for idx, snap in enumerate(quad_snapshots):
            if len(snap) == 0:
                continue
            src = snap[:, 0].astype(np.int64, copy=False)
            rel = snap[:, 1].astype(np.int64, copy=False)
            dst = snap[:, 2].astype(np.int64, copy=False)
            for s, r, o in zip(src, rel, dst):
                self.tail[(int(s), int(r))].append((idx, int(o)))
                self.rel[(int(s), int(o))].append((idx, int(r)))
        self.tail = {k: self._pack(v) for k, v in self.tail.items()}
        self.rel = {k: self._pack(v) for k, v in self.rel.items()}

    @staticmethod
    def _pack(values):
        arr = np.asarray(values, dtype=np.int64)
        if len(arr) > 1:
            order = np.argsort(arr[:, 0], kind="stable")
            arr = arr[order]
        return arr[:, 0], arr[:, 1]

    def make_vocab(self, torch, current_quad, history_limit, device):
        queries = np.asarray(current_quad, dtype=np.int64)
        tail_vocab = np.zeros((len(queries), self.num_nodes), dtype=np.float32)
        rel_vocab = np.zeros((len(queries), self.num_rels), dtype=np.float32)
        limit = int(history_limit)
        for row, (s, r, o, _) in enumerate(queries):
            packed = self.tail.get((int(s), int(r)))
            if packed is not None:
                idxs, vals = packed
                end = np.searchsorted(idxs, limit, side="left")
                if end:
                    tail_vocab[row, vals[:end]] = 1.0
            packed = self.rel.get((int(s), int(o)))
            if packed is not None:
                idxs, vals = packed
                end = np.searchsorted(idxs, limit, side="left")
                if end:
                    rel_vocab[row, vals[:end]] = 1.0

        tail_tensor = torch.from_numpy(tail_vocab).to(device)
        rel_tensor = torch.from_numpy(rel_vocab).to(device)
        return tail_tensor, rel_tensor


def make_out_dir(args):
    save_suffix = getattr(args, "save", "fair")
    save_suffix = "" if save_suffix in ("", "fair") else f"_{save_suffix}"
    name = (
        f"nsq{args.ns_q}_ns{args.ns_seed}_tpr{args.train_predict_ratio:g}"
        f"_h{args.n_hidden}_ly{args.n_layers}_his{args.train_history_len}"
        f"_hr{args.history_rate:g}_lr{args.lr:g}"
        f"{save_suffix}"
    )
    return osp.join("results_tirgn_fair", args.dataset, f"seed{args.seed}", name)


def load_static_graph(args, data, num_nodes, use_cuda, torch, build_sub_graph, read_triplets):
    if not args.add_static_graph:
        return 0, 0, None
    root = data.get("root") or getattr(data["dataset"], "root", "")
    path = osp.join(root, "e-w-graph.txt")
    if not osp.isfile(path):
        print(f"[TiRGN-Fair] static graph requested but missing {path}; disabling it.", flush=True)
        args.add_static_graph = False
        return 0, 0, None
    static_triples = np.asarray(read_triplets(path, {}, {}, load_time=False), dtype=np.int64)
    if len(static_triples) == 0:
        args.add_static_graph = False
        return 0, 0, None
    raw_static_rels = int(len(np.unique(static_triples[:, 1])))
    num_words = int(len(np.unique(static_triples[:, 2])))
    static_triples[:, 2] = static_triples[:, 2] + int(num_nodes)
    inverse = static_triples[:, [2, 1, 0]].copy()
    inverse[:, 1] += raw_static_rels
    static_triples = np.vstack((static_triples, inverse)).astype(np.int64, copy=False)
    num_static_rels = raw_static_rels * 2
    static_graph = build_sub_graph(num_nodes + num_words, num_static_rels, static_triples, use_cuda, args.gpu, add_inverse=False)
    return num_static_rels, num_words, static_graph


def build_model(
    args,
    data,
    train_quads,
    val_quads,
    test_quads,
    torch,
    RecurrentRGCN,
    build_sub_graph,
    read_triplets,
):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and int(args.gpu) >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    use_cuda = device.type == "cuda"
    num_nodes = int(data["num_nodes"])
    num_rels = int(data["num_rels"])
    all_quads = train_quads + val_quads + test_quads
    time_interval = infer_time_interval(all_quads)
    num_times = infer_num_times(all_quads, time_interval)
    print(
        f"[TiRGN-Fair] initializing model with time_interval={time_interval} "
        f"num_times={num_times} allocate_time_linears=0",
        flush=True,
    )
    num_static_rels, num_words, static_graph = load_static_graph(
        args, data, num_nodes, use_cuda, torch, build_sub_graph, read_triplets
    )
    model = RecurrentRGCN(
        args.decoder,
        args.encoder,
        num_nodes,
        num_rels,
        num_static_rels,
        num_words,
        num_times,
        time_interval,
        args.n_hidden,
        args.opn,
        args.history_rate,
        sequence_len=args.train_history_len,
        num_bases=args.n_bases,
        num_basis=args.n_basis,
        num_hidden_layers=args.n_layers,
        dropout=args.dropout,
        self_loop=args.self_loop,
        skip_connect=args.skip_connect,
        layer_norm=args.layer_norm,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout,
        aggregation=args.aggregation,
        weight=args.weight,
        discount=args.discount,
        angle=args.angle,
        use_static=args.add_static_graph,
        entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction,
        use_cuda=use_cuda,
        gpu=args.gpu if use_cuda else torch.device("cpu"),
        analysis=args.run_analysis,
        add_inverse=False,
        allocate_time_linears=False,
    )
    model.to(device)
    return model, static_graph, device, use_cuda, num_nodes, num_rels, time_interval, num_times


def build_history_graphs(quad_snapshots, start, end, num_nodes, num_rels, use_cuda, args, build_sub_graph):
    start = max(0, int(start))
    end = max(start, int(end))
    return [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu, add_inverse=False) for snap in quad_snapshots[start:end]]


def train_one_epoch(
    args,
    epoch,
    model,
    optimizer,
    scheduler,
    train_quads,
    history_index,
    static_graph,
    device,
    use_cuda,
    num_nodes,
    num_rels,
    torch,
    build_sub_graph,
):
    model.train()
    order = list(range(len(train_quads)))
    if args.shuffle_snapshots:
        random.shuffle(order)
    losses = []
    losses_e = []
    losses_r = []
    losses_static = []
    start_time = time.perf_counter()
    for snap_idx in order:
        if snap_idx == 0 or len(train_quads[snap_idx]) == 0:
            continue
        local_start = max(0, snap_idx - int(args.train_history_len))
        history_glist = build_history_graphs(
            train_quads, local_start, snap_idx, num_nodes, num_rels, use_cuda, args, build_sub_graph
        )
        output = torch.from_numpy(train_quads[snap_idx]).long().to(device)
        tail_vocab, rel_vocab = history_index.make_vocab(torch, train_quads[snap_idx], snap_idx, device)
        loss_e, loss_r, loss_static = model.get_loss(history_glist, output, static_graph, tail_vocab, rel_vocab, use_cuda)
        loss = args.task_weight * loss_e + (1.0 - args.task_weight) * loss_r + loss_static
        losses.append(float(loss.detach().cpu().item()))
        losses_e.append(float(loss_e.detach().cpu().item()))
        losses_r.append(float(loss_r.detach().cpu().item()))
        losses_static.append(float(loss_static.detach().cpu().item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None and args.scheduler_step == "batch":
            scheduler.step()
    if scheduler is not None and args.scheduler_step == "epoch":
        scheduler.step()
    sync_device(torch, device)
    elapsed = time.perf_counter() - start_time
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "epoch": int(epoch),
        "loss": mean(losses),
        "loss_entity": mean(losses_e),
        "loss_relation": mean(losses_r),
        "loss_static": mean(losses_static),
        "train_time_s": float(elapsed),
    }


def evaluate_split(
    args,
    model,
    split_name,
    eval_quads,
    eval_raw_times,
    all_quads,
    history_index,
    static_graph,
    device,
    use_cuda,
    num_nodes,
    num_rels,
    split_start_idx,
    torch,
    build_sub_graph,
    measure_forward=False,
):
    model.eval()
    sums = {}
    forward_time = 0.0
    sample_count = 0
    neg_sampler = args._negative_sampler
    with torch.no_grad():
        for local_idx, snap in enumerate(eval_quads):
            global_idx = int(split_start_idx + local_idx)
            local_start = max(0, global_idx - int(args.test_history_len))
            history_glist = build_history_graphs(
                all_quads, local_start, global_idx, num_nodes, num_rels, use_cuda, args, build_sub_graph
            )
            raw_t = int(eval_raw_times[local_idx])
            model_t = int(snap[0, 3]) if len(snap) else 0
            for batch, neg_arr, neg_mask in collect_eval_batch(snap[:, :3], raw_t, neg_sampler, split_name, args.eval_batch_size):
                if len(batch) == 0:
                    continue
                t_col = np.full((len(batch), 1), model_t, dtype=np.int64)
                batch_quad = np.hstack((batch.astype(np.int64, copy=False), t_col))
                tail_vocab, rel_vocab = history_index.make_vocab(torch, batch_quad, global_idx, device)
                batch_tensor = torch.from_numpy(batch_quad).long().to(device)
                if measure_forward:
                    sync_device(torch, device)
                    t0 = time.perf_counter()
                _, final_score, _ = model.predict(
                    history_glist, num_rels, static_graph, batch_tensor, tail_vocab, rel_vocab, use_cuda
                )
                if measure_forward:
                    sync_device(torch, device)
                    forward_time += time.perf_counter() - t0
                score = final_score
                pos_idx = torch.from_numpy(batch[:, 2].astype(np.int64, copy=False)).long().to(device)
                pos_scores = score.gather(1, pos_idx.view(-1, 1)).detach().cpu().numpy().astype(np.float32)
                neg_clean = neg_arr.copy()
                neg_clean[neg_clean < 0] = 0
                neg_idx = torch.from_numpy(neg_clean.astype(np.int64, copy=False)).long().to(device)
                neg_scores = score.gather(1, neg_idx).detach().cpu().numpy().astype(np.float32)
                batch_sums = compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask)
                add_metric_sums(sums, batch_sums)
                sample_count += int(len(batch))
    metrics = finalize_metric_sums(sums)
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    profile = {
        "forward_time_s": float(forward_time),
        "sample_count": int(sample_count),
    }
    return metrics, profile


def make_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "step":
        import torch

        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma)
        )
    raise ValueError(f"unsupported lr scheduler: {args.lr_scheduler}")


def run(args):
    torch, RecurrentRGCN, build_sub_graph, read_triplets = import_tirgn()
    set_random_seed(args.seed)
    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[TiRGN-Fair]")
    out_dir = make_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    save_config(out_dir, vars(args))

    train_quads = make_quad_snapshots(data["train_list"])
    val_quads = make_quad_snapshots(data["val_list"])
    test_quads = make_quad_snapshots(data["test_list"])
    val_raw_times = snapshot_raw_times(data["val_list"])
    test_raw_times = snapshot_raw_times(data["test_list"])
    all_quads = train_quads + val_quads + test_quads
    val_start = len(train_quads)
    test_start = len(train_quads) + len(val_quads)
    history_index = TemporalHistoryIndex(all_quads, data["num_nodes"], data["num_rels"])
    args._negative_sampler = data["negative_sampler"]

    model, static_graph, device, use_cuda, num_nodes, num_rels, time_interval, num_times = build_model(
        args,
        data,
        train_quads,
        val_quads,
        test_quads,
        torch,
        RecurrentRGCN,
        build_sub_graph,
        read_triplets,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(args, optimizer)

    print(
        f"[TiRGN-Fair] model nodes={num_nodes} rels={num_rels} "
        f"raw_rels={data['num_rels_raw']} internal_inverse=0 "
        f"time_interval={time_interval} num_times={num_times} device={device}",
        flush=True,
    )
    checkpoint_path = osp.join(out_dir, "best_model.pt")
    best_val = -float("inf")
    best_epoch = 0
    early_stopped = False
    early_stop_epoch = 0
    train_time_total = 0.0
    epoch_logs = []
    reset_cuda_peak(torch, device)
    for epoch in range(1, int(args.n_epochs) + 1):
        log = train_one_epoch(
            args,
            epoch,
            model,
            optimizer,
            scheduler,
            train_quads,
            history_index,
            static_graph,
            device,
            use_cuda,
            num_nodes,
            num_rels,
            torch,
            build_sub_graph,
        )
        train_time_total += log["train_time_s"]
        do_val = epoch > 1 and ((epoch - 1) % int(args.evaluate_every) == 0)
        if do_val:
            val_metrics, _ = evaluate_split(
                args,
                model,
                "val",
                val_quads,
                val_raw_times,
                all_quads,
                history_index,
                static_graph,
                device,
                use_cuda,
                num_nodes,
                num_rels,
                val_start,
                torch,
                build_sub_graph,
                measure_forward=False,
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
            f"[TiRGN-Fair] epoch={epoch} loss={log['loss']:.5f} "
            f"ent={log['loss_entity']:.5f} rel={log['loss_relation']:.5f} "
            f"static={log['loss_static']:.5f} train_time={log['train_time_s']:.2f}s "
            f"best_val_mrr={max(best_val, 0.0):.5f}",
            flush=True,
        )
        if do_val and best_epoch and int(epoch - best_epoch) >= int(args.patience):
            early_stopped = True
            early_stop_epoch = int(epoch)
            print(
                f"[TiRGN-Fair] early stop at epoch={epoch}: "
                f"val_mrr did not improve for {epoch - best_epoch} epochs "
                f"(patience={args.patience}, best_epoch={best_epoch})",
                flush=True,
            )
            break

    train_peak = cuda_peak_allocated(torch, device)
    if osp.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        best_epoch = int(args.n_epochs)
        torch.save({"state_dict": model.state_dict(), "epoch": best_epoch, "val_metrics": {}}, checkpoint_path)

    reset_cuda_peak(torch, device)
    val_metrics, _ = evaluate_split(
        args,
        model,
        "val",
        val_quads,
        val_raw_times,
        all_quads,
        history_index,
        static_graph,
        device,
        use_cuda,
        num_nodes,
        num_rels,
        val_start,
        torch,
        build_sub_graph,
        measure_forward=False,
    )
    test_metrics, test_profile = evaluate_split(
        args,
        model,
        "test",
        test_quads,
        test_raw_times,
        all_quads,
        history_index,
        static_graph,
        device,
        use_cuda,
        num_nodes,
        num_rels,
        test_start,
        torch,
        build_sub_graph,
        measure_forward=True,
    )
    eval_peak = cuda_peak_allocated(torch, device)
    metrics = {
        "format": "tirgn_fair_v1",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "ns_q": int(args.ns_q),
        "ns_seed": int(args.ns_seed),
        "train_predict_ratio": float(args.train_predict_ratio),
        "best_epoch": int(best_epoch),
        "best_val_mrr": float(best_val),
        "early_stopped": bool(early_stopped),
        "early_stop_epoch": int(early_stop_epoch),
        "patience": int(args.patience),
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
            "TiRGN RecurrentRGCN/get_loss/predict/build_sub_graph run with add_inverse=False; "
            "all reverse-edge policy comes from EAGLE's shared dataset loader. "
            "final ranking is restricted to one positive plus the protocol negatives."
        ),
    }
    save_metrics(out_dir, metrics)
    print(
        f"[TiRGN-Fair] final val_mrr={metrics['val_mrr']:.6f} "
        f"test_mrr={metrics['test_mrr']:.6f} test_hit1={metrics['test_hit1']:.6f} "
        f"test_hit10={metrics['test_hit10']:.6f}",
        flush=True,
    )
    print(
        f"[TiRGN-Fair] train_time={train_time_total:.3f}s "
        f"test_forward_time={test_profile['forward_time_s']:.3f}s "
        f"test_samples={test_profile['sample_count']} "
        f"train_peak={format_bytes(train_peak)} eval_peak={format_bytes(eval_peak)}",
        flush=True,
    )
    print(f"[TiRGN-Fair] saved -> {out_dir}", flush=True)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser("Fair TiRGN trainer for EAGLE TKG/THG protocols.")
    parser.add_argument("--dataset", type=str, required=True, choices=eagle_utils.SUPPORTED_DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=256)

    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--run-analysis", action="store_true", default=False)
    parser.add_argument("--run-statistic", action="store_true", default=False)
    parser.add_argument("--multi-step", action="store_true", default=False)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--add-static-graph", action="store_true", default=False)
    parser.add_argument("--add-rel-word", action="store_true", default=False)
    parser.add_argument("--relation-evaluation", action="store_true", default=False)

    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--task-weight", type=float, default=0.7)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--angle", type=int, default=10)
    parser.add_argument("--encoder", type=str, default="convgcn")
    parser.add_argument("--aggregation", type=str, default="none")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--skip-connect", action="store_true", default=False)
    parser.add_argument("--n-hidden", type=int, default=200)
    parser.add_argument("--opn", type=str, default="sub")
    parser.add_argument("--n-bases", type=int, default=100)
    parser.add_argument("--n-basis", type=int, default=100)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--self-loop", action="store_true", default=True)
    parser.add_argument("--no-self-loop", dest="self_loop", action="store_false")
    parser.add_argument("--layer-norm", action="store_true", default=False)
    parser.add_argument("--relation-prediction", action="store_true", default=False)
    parser.add_argument("--entity-prediction", action="store_true", default=False)
    parser.add_argument("--split_by_relation", action="store_true", default=False)

    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=9999)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--lr-scheduler", type=str, default="none", choices=("none", "step"))
    parser.add_argument("--lr-step-size", type=int, default=50)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-step", type=str, default="epoch", choices=("epoch", "batch"))

    parser.add_argument("--decoder", type=str, default="timeconvtranse")
    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dropout", type=float, default=0.2)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--train-history-len", type=int, default=10)
    parser.add_argument("--test-history-len", type=int, default=10)
    parser.add_argument("--dilate-len", type=int, default=1)
    parser.add_argument("--grid-search", action="store_true", default=False)
    parser.add_argument("--tune", type=str, default="history_len,n_layers,dropout,n_bases,angle,history_rate")
    parser.add_argument("--num-k", type=int, default=500)
    parser.add_argument("--history-rate", type=float, default=0.3)
    parser.add_argument("--save", type=str, default="fair")
    parser.add_argument("--shuffle-snapshots", action="store_true", default=True)
    parser.add_argument("--no-shuffle-snapshots", dest="shuffle_snapshots", action="store_false")
    args = parser.parse_args()
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if int(args.evaluate_every) <= 0:
        raise ValueError("--evaluate-every must be positive")
    if int(args.n_epochs) <= 0:
        raise ValueError("--n-epochs must be positive")
    if int(args.patience) <= 0:
        raise ValueError("--patience must be positive")
    if int(args.n_hidden) <= 0:
        raise ValueError("--n-hidden must be positive")
    if int(args.n_bases) <= 0:
        raise ValueError("--n-bases must be positive")
    if bool(args.add_static_graph) and int(args.n_hidden) % int(args.n_bases) != 0:
        raise ValueError(
            "--add-static-graph uses TiRGN's RGCNBlockLayer, which requires "
            "--n-hidden to be divisible by --n-bases; got "
            f"n_hidden={args.n_hidden}, n_bases={args.n_bases}"
        )
    if int(args.train_history_len) <= 0 or int(args.test_history_len) <= 0:
        raise ValueError("--train-history-len and --test-history-len must be positive")
    if int(args.eval_batch_size) <= 0:
        raise ValueError("--eval_batch_size must be positive")
    if args.encoder == "convgcn" and int(args.n_layers) > 2:
        raise ValueError(
            "TiRGN's original convgcn implementation supports --n-layers <= 2: "
            "RecurrentRGCN.forward passes exactly two relation states to RGCNCell.forward."
        )
    return args


if __name__ == "__main__":
    run(parse_args())
