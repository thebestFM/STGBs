import argparse
import copy
import os
import os.path as osp
import sys
import time

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
TGB2_DIR = osp.join(REPO_DIR, "TGB2")


def ensure_tgb2_import_path():
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    if TGB2_DIR not in sys.path:
        sys.path.insert(0, TGB2_DIR)


def import_edgebank():
    ensure_tgb2_import_path()
    from TGB2.modules.edgebank_predictor import EdgeBankPredictor

    return EdgeBankPredictor


def flatten_snapshots(snapshot_list):
    chunks = []
    for events, _, t_orig in snapshot_list:
        if len(events) == 0:
            continue
        t_col = np.full((len(events), 1), int(t_orig), dtype=np.int64)
        chunks.append(np.hstack((events.astype(np.int64, copy=False), t_col)))
    if not chunks:
        return np.empty((0, 4), dtype=np.int64)
    return np.vstack(chunks).astype(np.int64, copy=False)


def make_out_dir(args):
    name = (
        f"nsq{args.ns_q}_ns{args.ns_seed}_tpr{args.train_predict_ratio:g}"
        f"_mem{args.mem_mode}_twr{args.time_window_ratio:g}_bs{args.eval_batch_size}"
    )
    return osp.join("results_edgebank_fair", args.dataset, f"seed{args.seed}", name)


def evaluate_split(args, predictor, split_name, snapshot_list, measure_forward=False):
    sums = {}
    forward_time = 0.0
    sample_count = 0
    neg_sampler = args._negative_sampler

    for events, _, raw_t in snapshot_list:
        if len(events) == 0:
            continue
        for batch, neg_arr, neg_mask in collect_eval_batch(
            events[:, :3], int(raw_t), neg_sampler, split_name, args.eval_batch_size
        ):
            if len(batch) == 0:
                continue
            pos_scores = np.zeros((len(batch), 1), dtype=np.float32)
            neg_scores = np.zeros_like(neg_arr, dtype=np.float32)

            for i, query in enumerate(batch):
                valid_len = int(np.sum(neg_mask[i]))
                candidates = np.concatenate((np.array([int(query[2])], dtype=np.int64), neg_arr[i, :valid_len]))
                query_src = np.full(len(candidates), int(query[0]), dtype=np.int64)

                if measure_forward:
                    t0 = time.perf_counter()
                scores = predictor.predict_link(query_src, candidates)
                if measure_forward:
                    forward_time += time.perf_counter() - t0

                pos_scores[i, 0] = float(scores[0])
                if valid_len:
                    neg_scores[i, :valid_len] = scores[1:].astype(np.float32, copy=False)

            add_metric_sums(sums, compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask))
            sample_count += int(len(batch))

            ts = np.full(len(batch), int(raw_t), dtype=np.int64)
            predictor.update_memory(batch[:, 0], batch[:, 2], ts)

    metrics = finalize_metric_sums(sums)
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    return metrics, {"forward_time_s": float(forward_time), "sample_count": int(sample_count)}


def run(args):
    EdgeBankPredictor = import_edgebank()
    set_random_seed(args.seed)
    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[EdgeBank-Fair]")
    args._negative_sampler = data["negative_sampler"]

    out_dir = make_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    save_config(out_dir, {k: v for k, v in vars(args).items() if not k.startswith("_")})

    train_events = flatten_snapshots(data["train_list"])
    t0 = time.perf_counter()
    predictor = EdgeBankPredictor(
        train_events[:, 0],
        train_events[:, 2],
        train_events[:, 3],
        memory_mode=args.mem_mode,
        time_window_ratio=args.time_window_ratio,
    )
    train_time = time.perf_counter() - t0

    val_metrics, val_profile = evaluate_split(args, predictor, "val", data["val_list"], measure_forward=False)
    test_metrics, test_profile = evaluate_split(args, predictor, "test", data["test_list"], measure_forward=True)

    metrics = {
        "format": "edgebank_fair_v1",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "ns_q": int(args.ns_q),
        "ns_seed": int(args.ns_seed),
        "train_predict_ratio": float(args.train_predict_ratio),
        "memory_mode": args.mem_mode,
        "time_window_ratio": float(args.time_window_ratio),
        "train_time_s": float(train_time),
        "train_peak_allocated_bytes": None,
        "eval_peak_allocated_bytes": None,
        "test_forward_time_s": float(test_profile["forward_time_s"]),
        "test_inference_sample_count": int(test_profile["sample_count"]),
        "val_forward_time_s": float(val_profile["forward_time_s"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_mrr": float(val_metrics["mrr_strict"]),
        "val_hit1": float(val_metrics["hit@1_strict"]),
        "val_hit10": float(val_metrics["hit@10_strict"]),
        "test_mrr": float(test_metrics["mrr_strict"]),
        "test_hit1": float(test_metrics["hit@1_strict"]),
        "test_hit10": float(test_metrics["hit@10_strict"]),
        "model_note": (
            "TGB2 EdgeBankPredictor is used on CPU. Memory is initialized from train edges "
            "and updated online after validation/test positive batches, matching the TGB2 example. "
            "Ranking uses one positive plus the protocol negatives."
        ),
    }
    save_metrics(out_dir, metrics)
    print(
        f"[EdgeBank-Fair] val_mrr={metrics['val_mrr']:.6f} test_mrr={metrics['test_mrr']:.6f} "
        f"test_hit1={metrics['test_hit1']:.6f} test_hit10={metrics['test_hit10']:.6f}",
        flush=True,
    )
    print(
        f"[EdgeBank-Fair] train_time={train_time:.3f}s test_forward_time={test_profile['forward_time_s']:.3f}s "
        f"test_samples={test_profile['sample_count']} saved -> {out_dir}",
        flush=True,
    )
    return metrics


def tuning(args):
    mem_modes = ["unlimited", "fixed_time_window"]
    time_window_ratios = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]
    results = []
    best = None
    for mem_mode in mem_modes:
        ratios = [-1.0] if mem_mode == "unlimited" else time_window_ratios
        for ratio in ratios:
            trial_args = copy.deepcopy(args)
            trial_args.tune_edgebank = False
            trial_args.mem_mode = mem_mode
            trial_args.time_window_ratio = float(ratio if mem_mode == "fixed_time_window" else 0.15)
            print(
                f"[EdgeBank-Tune] start mem_mode={trial_args.mem_mode} "
                f"time_window_ratio={trial_args.time_window_ratio:g}",
                flush=True,
            )
            metrics = run(trial_args)
            record = {
                "mem_mode": trial_args.mem_mode,
                "time_window_ratio": float(trial_args.time_window_ratio),
                "val_mrr": float(metrics["val_mrr"]),
                "val_hit1": float(metrics["val_hit1"]),
                "val_hit10": float(metrics["val_hit10"]),
                "test_mrr": float(metrics["test_mrr"]),
                "test_hit1": float(metrics["test_hit1"]),
                "test_hit10": float(metrics["test_hit10"]),
                "out_dir": make_out_dir(trial_args),
            }
            results.append(record)
            if best is None or record["val_mrr"] > best["val_mrr"]:
                best = record

    best_test = max(results, key=lambda item: item["test_mrr"]) if results else None

    tune_dir = osp.join("results_edgebank_fair", args.dataset, f"seed{args.seed}", "tuning")
    os.makedirs(tune_dir, exist_ok=True)
    summary = {
        "format": "edgebank_tuning_v1",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "ns_q": int(args.ns_q),
        "ns_seed": int(args.ns_seed),
        "train_predict_ratio": float(args.train_predict_ratio),
        "selection_metric": "val_mrr_strict",
        "best": best,
        "best_by_test_mrr": best_test,
        "results": results,
    }
    save_metrics(tune_dir, summary)
    print(
        f"[EdgeBank-Tune] best mem_mode={best['mem_mode']} "
        f"time_window_ratio={best['time_window_ratio']:g} "
        f"val_mrr={best['val_mrr']:.6f} test_mrr={best['test_mrr']:.6f} "
        f"saved -> {tune_dir}",
        flush=True,
    )
    print(
        f"[EdgeBank-Tune] best_by_test_mrr mem_mode={best_test['mem_mode']} "
        f"time_window_ratio={best_test['time_window_ratio']:g} "
        f"val_mrr={best_test['val_mrr']:.6f} "
        f"test_mrr={best_test['test_mrr']:.6f} "
        f"test_hit1={best_test['test_hit1']:.6f} "
        f"test_hit10={best_test['test_hit10']:.6f}",
        flush=True,
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser("Fair EdgeBank trainer/evaluator for EAGLE protocols.")
    parser.add_argument("--dataset", type=str, required=True, choices=eagle_utils.SUPPORTED_DATASETS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--eval_batch_size", type=int, default=200)
    parser.add_argument("--mem_mode", type=str, default="unlimited", choices=("unlimited", "fixed_time_window"))
    parser.add_argument("--time_window_ratio", type=float, default=0.15)
    parser.add_argument("--tune-edgebank", action="store_true", default=False)
    args = parser.parse_args()
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if int(args.eval_batch_size) <= 0:
        raise ValueError("--eval_batch_size must be positive")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.tune_edgebank:
        tuning(parsed_args)
    else:
        run(parsed_args)
