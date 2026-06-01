import argparse
import os
import os.path as osp
import sys
import time
from collections import Counter

import numpy as np

import utils as eagle_utils
from utils import (
    add_metric_sums,
    compute_ranking_metric_sums,
    describe_loaded_data,
    load_datasets,
    save_config,
    save_metrics,
    set_random_seed,
    finalize_metric_sums,
)


REPO_DIR = osp.dirname(osp.abspath(__file__))
TGB2_DIR = osp.join(REPO_DIR, "TGB2")


def ensure_tgb2_import_path():
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    if TGB2_DIR not in sys.path:
        sys.path.insert(0, TGB2_DIR)


def import_tgb2_helpers():
    ensure_tgb2_import_path()
    from TGB2.modules.tkg_utils import create_basis_dict

    return create_basis_dict


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


def group_by(data, key_idx):
    grouped = {}
    if len(data) == 0:
        return grouped
    order = np.argsort(data[:, key_idx], kind="stable")
    sorted_data = data[order]
    start = 0
    while start < len(sorted_data):
        key = int(sorted_data[start, key_idx])
        end = start + 1
        while end < len(sorted_data) and int(sorted_data[end, key_idx]) == key:
            end += 1
        grouped[key] = np.ascontiguousarray(sorted_data[start:end], dtype=np.int64)
        start = end
    return grouped


def quads_per_rel(quads):
    return group_by(quads, 1)


def make_out_dir(args):
    name = (
        f"nsq{args.ns_q}_ns{args.ns_seed}_tpr{args.train_predict_ratio:g}"
        f"_w{args.window}_a{args.alpha:g}_l{args.lmbda:g}"
        f"_train{int(args.train_flag)}_bs{args.eval_batch_size}"
    )
    return osp.join("results_recurrencybaseline_fair", args.dataset, f"seed{args.seed}", name)


def create_scores_array(predictions_dict, num_nodes):
    predictions = np.zeros(int(num_nodes), dtype=np.float64)
    if not predictions_dict:
        return predictions
    keys = np.asarray(list(predictions_dict.keys()), dtype=np.int64)
    values = np.asarray(list(predictions_dict.values()), dtype=np.float64)
    valid = (keys >= 0) & (keys < int(num_nodes))
    predictions[keys[valid]] = values[valid]
    return predictions


def score_delta(cands_ts, test_query_ts, lmbda):
    return np.power(2.0, float(lmbda) * (np.asarray(cands_ts) - float(test_query_ts)))


def update_delta_t(min_ts, max_ts, cur_ts, lmbda):
    timesteps = np.arange(int(min_ts), int(max_ts), dtype=np.float64)
    if len(timesteps) == 0:
        return 0.0
    return float(np.sum(score_delta(timesteps, cur_ts, lmbda)))


def get_window_edges(all_data, test_query_ts, window=-2, first_test_query_ts=0):
    if len(all_data) == 0:
        return {}, np.empty((0, all_data.shape[1] if all_data.ndim == 2 else 5), dtype=np.int64)
    if int(window) > 0:
        mask = (all_data[:, 3] < int(test_query_ts)) & (all_data[:, 3] >= int(test_query_ts) - int(window))
    elif int(window) == 0:
        mask = all_data[:, 3] < int(test_query_ts)
    elif int(window) == -2:
        mask = all_data[:, 3] < int(first_test_query_ts)
    elif int(window) == -200:
        mask = (all_data[:, 3] < int(first_test_query_ts)) & (all_data[:, 3] >= int(first_test_query_ts) - 200)
    else:
        raise ValueError("--window must be one of -2, -200, 0, or a positive integer")
    selected = np.ascontiguousarray(all_data[mask], dtype=np.int64)
    return quads_per_rel(selected), selected


def calculate_obj_distribution(edges, num_rels):
    rel_obj_dist = {rel: {} for rel in range(int(num_rels))}
    for rel, rel_edges in edges.items():
        objects = rel_edges[:, 2]
        dist = Counter(int(x) for x in objects)
        denom = float(len(objects)) if len(objects) else 1.0
        rel_obj_dist[int(rel)] = {int(obj): count / denom for obj, count in dist.items()}
    return rel_obj_dist


def match_body_relations(rule, edges, test_query_sub):
    rels = rule["body_rels"]
    try:
        rel_edges = edges[int(rels[0])]
        mask = rel_edges[:, 0] == int(test_query_sub)
        new_edges = rel_edges[mask]
        return [np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))]
    except KeyError:
        return [[]]


def score_psi(cands_walks, test_query_ts, lmbda, sum_delta_t):
    all_cands_ts = cands_walks[:, 1]
    scores = score_delta(all_cands_ts, test_query_ts, lmbda)
    if float(sum_delta_t) == 0.0:
        return float(np.sum(scores))
    return float(np.sum(scores) / float(sum_delta_t))


def get_candidates_psi(rule_walks, test_query_ts, cands_dict, lmbda, sum_delta_t):
    cands = set(int(x) for x in rule_walks[:, 0])
    for cand in cands:
        cands_walks = rule_walks[rule_walks[:, 0] == cand]
        cands_dict[cand] = score_psi(cands_walks, test_query_ts, lmbda, sum_delta_t)
    return cands_dict


def rb_scores_for_queries(args, queries, all_data, basis_dict, num_nodes, num_rels, first_query_ts):
    pos_scores = np.zeros((len(queries), 1), dtype=np.float32)
    neg_lists = []
    neg_sampler = args._negative_sampler
    forward_time = 0.0

    if len(queries) == 0:
        return pos_scores, np.zeros((0, 1), dtype=np.float32), np.zeros((0, 1), dtype=bool), forward_time

    cur_ts = int(queries[0, 3])
    edges, all_data_ts = get_window_edges(all_data, cur_ts, args.window, first_query_ts)
    rel_obj_dist_cur_ts = calculate_obj_distribution(edges, num_rels)
    sum_delta_t = 0.0
    if len(all_data_ts) > 0:
        sum_delta_t = update_delta_t(np.min(all_data_ts[:, 3]), np.max(all_data_ts[:, 3]), cur_ts, args.lmbda)

    pos_values = []
    neg_values = []
    max_negs = 1
    for row, query in enumerate(queries):
        if int(query[3]) != cur_ts:
            cur_ts = int(query[3])
            edges, all_data_ts = get_window_edges(all_data, cur_ts, args.window, first_query_ts)
            if int(args.window) > -1:
                rel_obj_dist_cur_ts = calculate_obj_distribution(edges, num_rels)
                if len(all_data_ts) > 0:
                    sum_delta_t = update_delta_t(np.min(all_data_ts[:, 3]), np.max(all_data_ts[:, 3]), cur_ts, args.lmbda)

        neg = neg_sampler.query_batch(
            np.asarray([int(query[0])], dtype=np.int64),
            np.asarray([int(query[2])], dtype=np.int64),
            np.asarray([int(query[4])], dtype=np.int64),
            np.asarray([int(query[1])], dtype=np.int64),
            args._split_name,
        )[0]
        neg = np.asarray(neg, dtype=np.int64)
        max_negs = max(max_negs, len(neg))

        t0 = time.perf_counter()
        cands_dict_psi = {}
        if str(int(query[1])) in basis_dict:
            walk_edges = match_body_relations(basis_dict[str(int(query[1]))][0], edges, int(query[0]))
            if 0 not in [len(x) for x in walk_edges]:
                cands_dict_psi = get_candidates_psi(walk_edges[0][:, 1:3], cur_ts, {}, args.lmbda, sum_delta_t)
        predictions_psi = create_scores_array(cands_dict_psi, num_nodes)
        predictions_xi = create_scores_array(rel_obj_dist_cur_ts.get(int(query[1]), {}), num_nodes)
        predictions_all = 1000.0 * float(args.alpha) * predictions_psi + 1000.0 * (1.0 - float(args.alpha)) * predictions_xi
        forward_time += time.perf_counter() - t0

        pos_values.append(float(predictions_all[int(query[2])]))
        neg_values.append(predictions_all[neg].astype(np.float32, copy=False))
        neg_lists.append(neg)

    neg_scores = np.zeros((len(queries), max_negs), dtype=np.float32)
    neg_mask = np.zeros((len(queries), max_negs), dtype=bool)
    for i, values in enumerate(neg_values):
        if len(values):
            neg_scores[i, : len(values)] = values
            neg_mask[i, : len(values)] = True
    pos_scores[:, 0] = np.asarray(pos_values, dtype=np.float32)
    return pos_scores, neg_scores, neg_mask, forward_time


def evaluate_events(args, events, all_data, basis_dict, num_nodes, num_rels, split_name, measure_forward=False):
    sums = {}
    sample_count = 0
    forward_time = 0.0
    if len(events) == 0:
        return finalize_metric_sums(sums), {"forward_time_s": 0.0, "sample_count": 0}
    args._split_name = split_name
    first_query_ts = int(events[0, 3])
    for start in range(0, len(events), int(args.eval_batch_size)):
        batch = np.ascontiguousarray(events[start : start + int(args.eval_batch_size)], dtype=np.int64)
        pos_scores, neg_scores, neg_mask, elapsed = rb_scores_for_queries(
            args, batch, all_data, basis_dict, num_nodes, num_rels, first_query_ts
        )
        if measure_forward:
            forward_time += elapsed
        add_metric_sums(sums, compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask))
        sample_count += int(len(batch))
    metrics = finalize_metric_sums(sums)
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    return metrics, {"forward_time_s": float(forward_time), "sample_count": int(sample_count)}


def tune_params(args, rels, val_by_rel, trainval_by_rel, basis_dict, num_nodes, num_rels):
    lmbdas = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.5, 0.9, 1.0001]
    alphas = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1]
    best_config = {}
    original_lambda = args.lmbda
    original_alpha = args.alpha
    for rel in rels:
        rel = int(rel)
        best_config[str(rel)] = {
            "not_trained": "True",
            "lmbda_psi": [float(lmbdas[-1]), 0.0],
            "alpha": [float(alphas[-2]), 0.0],
            "other_lmbda_mrrs": [0.0 for _ in lmbdas],
            "other_alpha_mrrs": [0.0 for _ in alphas],
        }
        if rel not in val_by_rel:
            continue
        best_config[str(rel)]["not_trained"] = "False"
        val_rel = val_by_rel[rel]
        trainval_rel = trainval_by_rel.get(rel, np.empty((0, 5), dtype=np.int64))

        args.alpha = 1.0
        lmbda_mrrs = []
        best_lmbda = 0.1
        best_lmbda_mrr = 0.0
        for lmbda in lmbdas:
            args.lmbda = float(lmbda)
            metrics, _ = evaluate_events(args, val_rel, trainval_rel, basis_dict, num_nodes, num_rels, "val", False)
            score = float(metrics["mrr_strict"])
            lmbda_mrrs.append(score)
            if score > best_lmbda_mrr:
                best_lmbda_mrr = score
                best_lmbda = float(lmbda)
        best_config[str(rel)]["lmbda_psi"] = [best_lmbda, best_lmbda_mrr]
        best_config[str(rel)]["other_lmbda_mrrs"] = lmbda_mrrs

        args.lmbda = best_lmbda
        alpha_mrrs = []
        best_alpha = 0.99
        best_alpha_mrr = 0.0
        for alpha in alphas:
            args.alpha = float(alpha)
            metrics, _ = evaluate_events(args, val_rel, trainval_rel, basis_dict, num_nodes, num_rels, "val", False)
            score = float(metrics["mrr_strict"])
            alpha_mrrs.append(score)
            if score > best_alpha_mrr:
                best_alpha_mrr = score
                best_alpha = float(alpha)
        best_config[str(rel)]["alpha"] = [best_alpha, best_alpha_mrr]
        best_config[str(rel)]["other_alpha_mrrs"] = alpha_mrrs

    args.lmbda = original_lambda
    args.alpha = original_alpha
    return best_config


def evaluate_by_relation(args, best_config, rels, split_by_rel, history_by_rel, basis_dict, num_nodes, num_rels, split_name, measure_forward):
    sums = {}
    total_forward = 0.0
    sample_count = 0
    original_lambda = args.lmbda
    original_alpha = args.alpha
    for rel in rels:
        rel = int(rel)
        if rel not in split_by_rel:
            continue
        cfg = best_config[str(rel)]
        args.lmbda = float(cfg["lmbda_psi"][0])
        args.alpha = float(cfg["alpha"][0])
        metrics, profile = evaluate_events(
            args,
            split_by_rel[rel],
            history_by_rel.get(rel, np.empty((0, 5), dtype=np.int64)),
            basis_dict,
            num_nodes,
            num_rels,
            split_name,
            measure_forward,
        )
        count = int(profile["sample_count"])
        add_metric_sums(sums, {k: v * count for k, v in metrics.items() if k.startswith("mrr_") or k.startswith("hit@")})
        sums["count"] = sums.get("count", 0.0) + count
        total_forward += float(profile["forward_time_s"])
        sample_count += count
    args.lmbda = original_lambda
    args.alpha = original_alpha
    metrics = finalize_metric_sums(sums)
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    return metrics, {"forward_time_s": float(total_forward), "sample_count": int(sample_count)}


def run(args):
    create_basis_dict = import_tgb2_helpers()
    set_random_seed(args.seed)
    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[RB-Fair]")
    args._negative_sampler = data["negative_sampler"]

    out_dir = make_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    save_config(out_dir, {k: v for k, v in vars(args).items() if not k.startswith("_")})

    train_events = flatten_snapshots(data["train_list"])
    val_events = flatten_snapshots(data["val_list"])
    test_events = flatten_snapshots(data["test_list"])
    trainval_events = np.vstack((train_events, val_events)) if len(val_events) else train_events.copy()
    all_events = np.vstack((trainval_events, test_events)) if len(test_events) else trainval_events.copy()
    num_nodes = int(data["num_nodes"])
    num_rels = int(data["num_rels"])
    rels = np.arange(num_rels, dtype=np.int64)

    t0 = time.perf_counter()
    basis_dict = create_basis_dict(trainval_events[:, :4]) if len(trainval_events) else {}
    val_by_rel = group_by(val_events, 1)
    test_by_rel = group_by(test_events, 1)
    trainval_by_rel = group_by(trainval_events, 1)
    all_by_rel = group_by(all_events, 1)

    if args.train_flag:
        best_config = tune_params(args, rels, val_by_rel, trainval_by_rel, basis_dict, num_nodes, num_rels)
    else:
        best_config = {
            str(int(rel)): {"lmbda_psi": [float(args.lmbda)], "alpha": [float(args.alpha)], "not_trained": "False"}
            for rel in rels
        }
    train_time = time.perf_counter() - t0

    val_metrics, val_profile = evaluate_by_relation(
        args, best_config, rels, val_by_rel, trainval_by_rel, basis_dict, num_nodes, num_rels, "val", False
    )
    test_metrics, test_profile = evaluate_by_relation(
        args, best_config, rels, test_by_rel, all_by_rel, basis_dict, num_nodes, num_rels, "test", True
    )

    metrics = {
        "format": "recurrencybaseline_fair_v1",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "ns_q": int(args.ns_q),
        "ns_seed": int(args.ns_seed),
        "train_predict_ratio": float(args.train_predict_ratio),
        "window": int(args.window),
        "train_flag": bool(args.train_flag),
        "default_lmbda": float(args.lmbda),
        "default_alpha": float(args.alpha),
        "best_config": best_config,
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
            "TGB2 RecurrencyBaseline psi/xi scoring is adapted to EAGLE data. "
            "No GPU is used. Ranking uses one positive plus the protocol negatives."
        ),
    }
    save_metrics(out_dir, metrics)
    print(
        f"[RB-Fair] val_mrr={metrics['val_mrr']:.6f} test_mrr={metrics['test_mrr']:.6f} "
        f"test_hit1={metrics['test_hit1']:.6f} test_hit10={metrics['test_hit10']:.6f}",
        flush=True,
    )
    print(
        f"[RB-Fair] train_time={train_time:.3f}s test_forward_time={test_profile['forward_time_s']:.3f}s "
        f"test_samples={test_profile['sample_count']} saved -> {out_dir}",
        flush=True,
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser("Fair RecurrencyBaseline trainer/evaluator for EAGLE protocols.")
    parser.add_argument("--dataset", type=str, required=True, choices=eagle_utils.SUPPORTED_DATASETS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--eval_batch_size", type=int, default=200)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--lmbda", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--train_flag", action="store_true", default=False)
    args = parser.parse_args()
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if int(args.eval_batch_size) <= 0:
        raise ValueError("--eval_batch_size must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
