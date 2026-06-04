import argparse
import json
import os
import os.path as osp
import sys
import time
import itertools

import numpy as np

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


def ensure_tgb2_import_path():
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    if TGB2_DIR not in sys.path:
        sys.path.insert(0, TGB2_DIR)


def import_tlogic():
    ensure_tgb2_import_path()
    from TGB2.modules.tlogic_learn_modules import Temporal_Walk, Rule_Learner, store_edges
    import TGB2.modules.tlogic_apply_modules as ra
    from TGB2.modules.tkg_utils import get_inv_relation_id, create_scores_array

    return Temporal_Walk, Rule_Learner, store_edges, ra, get_inv_relation_id, create_scores_array


def flatten_snapshots(snapshot_list, include_raw=True):
    chunks = []
    for events, t_norm, t_orig in snapshot_list:
        if len(events) == 0:
            continue
        t_norm_col = np.full((len(events), 1), int(t_norm), dtype=np.int64)
        if include_raw:
            t_orig_col = np.full((len(events), 1), int(t_orig), dtype=np.int64)
            chunks.append(np.hstack((events.astype(np.int64, copy=False), t_norm_col, t_orig_col)))
        else:
            chunks.append(np.hstack((events.astype(np.int64, copy=False), t_norm_col)))
    width = 5 if include_raw else 4
    if not chunks:
        return np.empty((0, width), dtype=np.int64)
    return np.vstack(chunks).astype(np.int64, copy=False)


def make_out_dir(args):
    lengths = "-".join(str(int(x)) for x in args.rule_lengths)
    name = (
        f"nsq{args.ns_q}_ns{args.ns_seed}_tpr{args.train_predict_ratio:g}"
        f"_r{lengths}_n{args.num_walks}_{args.transition_distr}"
        f"_w{args.window}_top{args.top_k}_l{args.score_lambda:g}_a{args.score_alpha:g}"
    )
    return osp.join("results_tlogic_fair", args.dataset, f"seed{args.seed}", name)


def split_chunks(items, num_chunks):
    items = list(items)
    num_chunks = max(1, int(num_chunks))
    if not items:
        return []
    size = int(np.ceil(len(items) / float(num_chunks)))
    return [items[i : i + size] for i in range(0, len(items), size)]


def learn_rule_chunk(
    relation_chunk,
    train_data,
    inv_relation_id,
    transition_distr,
    rule_lengths,
    num_walks,
    output_dir,
):
    Temporal_Walk, Rule_Learner, _, _, _, _ = import_tlogic()
    print(
        f"[TLogic-Fair] learn chunk: build Temporal_Walk relations={len(relation_chunk)} "
        f"events={len(train_data)}",
        flush=True,
    )
    t_tw = time.perf_counter()
    temporal_walk = Temporal_Walk(train_data, inv_relation_id, transition_distr)
    print(
        f"[TLogic-Fair] learn chunk: Temporal_Walk ready relations={len(temporal_walk.edges)} "
        f"elapsed={time.perf_counter() - t_tw:.1f}s",
        flush=True,
    )
    learner = Rule_Learner(
        edges=temporal_walk.edges,
        id2relation=None,
        inv_relation_id=inv_relation_id,
        output_dir=output_dir,
    )
    for rel in relation_chunk:
        for length in rule_lengths:
            t0 = time.perf_counter()
            before = sum(len(v) for v in learner.rules_dict.values())
            total_walks = int(num_walks)
            progress_step = max(1, int(np.ceil(total_walks / 100.0)))
            next_progress = progress_step
            successful_walks = 0
            for walk_idx in range(1, total_walks + 1):
                walk_successful, walk = temporal_walk.sample_walk(int(length) + 1, int(rel))
                if walk_successful:
                    successful_walks += 1
                    learner.create_rule(walk)
                while walk_idx >= next_progress or walk_idx == total_walks:
                    current_rules = sum(len(v) for v in learner.rules_dict.values())
                    print(
                        f"[TLogic-Fair] learn relation={rel} length={length}: "
                        f"walks {walk_idx}/{total_walks} "
                        f"({100.0 * walk_idx / max(1, total_walks):.1f}%) "
                        f"successful={successful_walks} new_rules={current_rules - before} "
                        f"elapsed={time.perf_counter() - t0:.1f}s",
                        flush=True,
                    )
                    if walk_idx >= total_walks:
                        break
                    next_progress += progress_step
            after = sum(len(v) for v in learner.rules_dict.values())
            print(
                f"[TLogic-Fair] relation={rel} length={length} "
                f"time={time.perf_counter() - t0:.3f}s new_rules={after - before}",
                flush=True,
            )
    return learner.rules_dict


def merge_rules(rule_dicts):
    merged = {}
    seen = set()
    for rules_dict in rule_dicts:
        for rel, rules in rules_dict.items():
            rel = int(rel)
            merged.setdefault(rel, [])
            for rule in rules:
                key = json.dumps(rule, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                merged[rel].append(rule)
    return merged


def save_rules_json(path, rules_dict):
    serializable = {str(int(k)): v for k, v in rules_dict.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)


def load_rules_json(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(k): v for k, v in payload.items()}


def resolve_rule_path(rule_filename, out_dir):
    if osp.isabs(rule_filename):
        return rule_filename
    local = osp.join(out_dir, rule_filename)
    if osp.exists(local):
        return local
    return rule_filename


def learn_or_load_rules(args, train_data, num_rels, out_dir):
    _, _, _, ra, get_inv_relation_id, _ = import_tlogic()
    inv_relation_id = get_inv_relation_id(num_rels)
    rules_dir = osp.join(out_dir, "rules")
    os.makedirs(rules_dir, exist_ok=True)

    if args.learn_rules_flag:
        if len(train_data) == 0:
            return {}, "", 0.0
        Temporal_Walk, _, _, _, _, _ = import_tlogic()
        print(
            f"[TLogic-Fair] learn rules: build relation index events={len(train_data)}",
            flush=True,
        )
        t_tw = time.perf_counter()
        temporal_walk = Temporal_Walk(train_data, inv_relation_id, args.transition_distr)
        print(
            f"[TLogic-Fair] learn rules: relation index ready relations={len(temporal_walk.edges)} "
            f"elapsed={time.perf_counter() - t_tw:.1f}s",
            flush=True,
        )
        all_relations = sorted(temporal_walk.edges)
        chunks = split_chunks(all_relations, args.num_processes)
        t0 = time.perf_counter()
        if int(args.num_processes) == 1:
            outputs = [
                learn_rule_chunk(
                    chunks[0],
                    train_data,
                    inv_relation_id,
                    args.transition_distr,
                    args.rule_lengths,
                    args.num_walks,
                    rules_dir,
                )
            ]
        else:
            from joblib import Parallel, delayed

            outputs = Parallel(n_jobs=int(args.num_processes))(
                delayed(learn_rule_chunk)(
                    chunk,
                    train_data,
                    inv_relation_id,
                    args.transition_distr,
                    args.rule_lengths,
                    args.num_walks,
                    rules_dir,
                )
                for chunk in chunks
            )
        rules_dict = merge_rules(outputs)
        # Reuse TLogic's rule sorting behavior before saving/filtering.
        _, Rule_Learner, _, _, _, _ = import_tlogic()
        learner = Rule_Learner(
            edges=temporal_walk.edges,
            id2relation=None,
            inv_relation_id=inv_relation_id,
            output_dir=rules_dir,
        )
        learner.rules_dict = rules_dict
        learner.sort_rules_dict()
        rules_dict = learner.rules_dict
        rule_filename = f"fair_r{args.rule_lengths}_n{args.num_walks}_{args.transition_distr}_s{args.seed}_rules.json"
        rule_filename = rule_filename.replace(" ", "")
        rule_path = osp.join(rules_dir, rule_filename)
        save_rules_json(rule_path, rules_dict)
        train_time = time.perf_counter() - t0
    else:
        rule_path = resolve_rule_path(args.rule_filename, out_dir)
        rules_dict = load_rules_json(rule_path)
        train_time = 0.0

    t_filter = time.perf_counter()
    rules_dict = ra.filter_rules(
        rules_dict,
        min_conf=float(args.min_conf),
        min_body_supp=int(args.min_body_supp),
        rule_lengths=[int(x) for x in args.rule_lengths],
    )
    train_time += time.perf_counter() - t_filter
    return rules_dict, rule_path, float(train_time)


def query_negatives(neg_sampler, split_name, query):
    return np.asarray(
        neg_sampler.query_batch(
            np.asarray([int(query[0])], dtype=np.int64),
            np.asarray([int(query[2])], dtype=np.int64),
            np.asarray([int(query[4])], dtype=np.int64),
            np.asarray([int(query[1])], dtype=np.int64),
            split_name,
        )[0],
        dtype=np.int64,
    )


def candidate_scores_for_query(args, query, rules_dict, edges, ra):
    cands_dict = [dict() for _ in range(1)]
    if int(query[1]) not in rules_dict:
        return {}

    dicts_idx = [0]
    for rule in rules_dict[int(query[1])]:
        walk_edges = ra.match_body_relations(rule, edges, int(query[0]))
        if 0 in [len(x) for x in walk_edges]:
            continue
        rule_walks = ra.get_walks(rule, walk_edges)
        if rule["var_constraints"]:
            rule_walks = ra.check_var_constraints(rule["var_constraints"], rule_walks)
        if rule_walks.empty:
            continue
        cands_dict = ra.get_candidates(
            rule,
            rule_walks,
            int(query[3]),
            cands_dict,
            ra.score_12,
            [[float(args.score_lambda), float(args.score_alpha)]],
            dicts_idx,
        )
        for s in list(dicts_idx):
            cands_dict[s] = {
                x: sorted(cands_dict[s][x], reverse=True)
                for x in cands_dict[s].keys()
            }
            cands_dict[s] = dict(
                sorted(cands_dict[s].items(), key=lambda item: item[1], reverse=True)
            )
            top_k_scores = [v for _, v in cands_dict[s].items()][: int(args.top_k)]
            unique_scores = list(scores for scores, _ in itertools.groupby(top_k_scores))
            if len(unique_scores) >= int(args.top_k):
                dicts_idx.remove(s)
        if not dicts_idx:
            break

    if not cands_dict[0]:
        return {}
    scores = [1.0 - np.prod(1.0 - np.asarray(v, dtype=np.float64)) for v in cands_dict[0].values()]
    return dict(
        sorted(dict(zip(cands_dict[0].keys(), scores)).items(), key=lambda x: x[1], reverse=True)
    )


def lookup_scores(candidate_scores, node_ids):
    return np.asarray([float(candidate_scores.get(int(node_id), 0.0)) for node_id in node_ids], dtype=np.float32)


def evaluate_event_chunk(
    args,
    events,
    rules_dict,
    learn_edges,
    all_data,
    num_nodes,
    split_name,
    measure_forward=False,
    process_id=0,
    split_first_query_ts=None,
):
    _, _, _, ra, _, _ = import_tlogic()
    sums = {}
    sample_count = 0
    forward_time = 0.0
    if len(events) == 0:
        return {"sums": sums, "forward_time_s": 0.0, "sample_count": 0, "wall_time_s": 0.0}

    neg_sampler = args._negative_sampler
    first_query_ts = int(split_first_query_ts if split_first_query_ts is not None else events[0, 3])
    cur_ts = None
    edges = {}
    wall_t0 = time.perf_counter()
    next_progress = max(1, int(np.ceil(len(events) / 100.0)))
    progress_step = next_progress

    for start in range(0, len(events), int(args.eval_batch_size)):
        batch = np.ascontiguousarray(events[start : start + int(args.eval_batch_size)], dtype=np.int64)
        neg_lists = [query_negatives(neg_sampler, split_name, query) for query in batch]
        width = max((len(x) for x in neg_lists), default=0) or 1
        pos_scores = np.zeros((len(batch), 1), dtype=np.float32)
        neg_scores = np.zeros((len(batch), width), dtype=np.float32)
        neg_mask = np.zeros((len(batch), width), dtype=bool)

        for i, query in enumerate(batch):
            if int(query[3]) != cur_ts:
                cur_ts = int(query[3])
                t0 = time.perf_counter()
                print(
                    f"[TLogic-Fair] eval {split_name} process={process_id}: "
                    f"build window edges ts={cur_ts}",
                    flush=True,
                )
                edges = ra.get_window_edges(
                    all_data[:, :4],
                    cur_ts,
                    learn_edges,
                    int(args.window),
                    first_test_query_ts=first_query_ts,
                )
                print(
                    f"[TLogic-Fair] eval {split_name} process={process_id}: "
                    f"window edges ready ts={cur_ts} relations={len(edges)} "
                    f"elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
                if measure_forward:
                    forward_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            candidate_scores = candidate_scores_for_query(args, query, rules_dict, edges, ra)
            if measure_forward:
                forward_time += time.perf_counter() - t0

            neg = neg_lists[i]
            pos_scores[i, 0] = float(candidate_scores.get(int(query[2]), 0.0))
            if len(neg):
                neg_scores[i, : len(neg)] = lookup_scores(candidate_scores, neg)
                neg_mask[i, : len(neg)] = True

        add_metric_sums(sums, compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask))
        sample_count += int(len(batch))
        while sample_count >= next_progress or sample_count == len(events):
            print(
                f"[TLogic-Fair] eval {split_name} process={process_id}: "
                f"completed {sample_count}/{len(events)} "
                f"({100.0 * sample_count / max(1, len(events)):.1f}%) "
                f"elapsed={time.perf_counter() - wall_t0:.1f}s",
                flush=True,
            )
            if sample_count >= len(events):
                break
            next_progress += progress_step

    return {
        "sums": sums,
        "forward_time_s": float(forward_time),
        "sample_count": int(sample_count),
        "wall_time_s": float(time.perf_counter() - wall_t0),
    }


def split_event_chunks(events, num_chunks):
    if len(events) == 0:
        return []
    num_chunks = max(1, min(int(num_chunks), len(events)))
    return [np.ascontiguousarray(chunk, dtype=np.int64) for chunk in np.array_split(events, num_chunks) if len(chunk)]


def evaluate_events(
    args,
    events,
    rules_dict,
    learn_edges,
    all_data,
    num_nodes,
    split_name,
    measure_forward=False,
):
    sums = {}
    if len(events) == 0:
        metrics = finalize_metric_sums(sums)
        metrics["mrr"] = metrics["mrr_strict"]
        metrics["hit1"] = metrics["hit@1_strict"]
        metrics["hit10"] = metrics["hit@10_strict"]
        return metrics, {"forward_time_s": 0.0, "sample_count": 0, "wall_time_s": 0.0}

    eval_t0 = time.perf_counter()
    split_first_query_ts = int(events[0, 3])
    chunks = split_event_chunks(np.ascontiguousarray(events, dtype=np.int64), int(args.num_processes))
    print(
        f"[TLogic-Fair] eval {split_name}: start queries={len(events)} "
        f"chunks={len(chunks)} processes={args.num_processes} ns_q={args.ns_q}",
        flush=True,
    )

    if int(args.num_processes) == 1 or len(chunks) == 1:
        outputs = [
            evaluate_event_chunk(
                args,
                chunks[0],
                rules_dict,
                learn_edges,
                all_data,
                num_nodes,
                split_name,
                measure_forward=measure_forward,
                process_id=0,
                split_first_query_ts=split_first_query_ts,
            )
        ]
    else:
        from joblib import Parallel, delayed

        outputs = Parallel(n_jobs=int(args.num_processes))(
            delayed(evaluate_event_chunk)(
                args,
                chunk,
                rules_dict,
                learn_edges,
                all_data,
                num_nodes,
                split_name,
                measure_forward=measure_forward,
                process_id=i,
                split_first_query_ts=split_first_query_ts,
            )
            for i, chunk in enumerate(chunks)
        )

    forward_time = 0.0
    sample_count = 0
    worker_wall_times = []
    for output in outputs:
        add_metric_sums(sums, output["sums"])
        forward_time += float(output["forward_time_s"])
        sample_count += int(output["sample_count"])
        worker_wall_times.append(float(output["wall_time_s"]))

    metrics = finalize_metric_sums(sums)
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    wall_time = time.perf_counter() - eval_t0
    print(
        f"[TLogic-Fair] eval {split_name}: done mrr={metrics['mrr_strict']:.6f} "
        f"hit1={metrics['hit@1_strict']:.6f} hit10={metrics['hit@10_strict']:.6f} "
        f"samples={sample_count} wall={wall_time:.1f}s forward_sum={forward_time:.1f}s",
        flush=True,
    )
    return metrics, {
        "forward_time_s": float(forward_time),
        "sample_count": int(sample_count),
        "wall_time_s": float(wall_time),
        "worker_wall_time_s": worker_wall_times,
    }


def run(args):
    _, _, store_edges, _, _, _ = import_tlogic()
    set_random_seed(args.seed)
    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[TLogic-Fair]")
    if data.get("is_thg"):
        raise ValueError("TLogic-Fair is adapted for TKG datasets only; Yelp-* THG datasets are unsupported here.")

    out_dir = make_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    save_config(out_dir, {k: v for k, v in vars(args).items() if not k.startswith("_")})
    args._negative_sampler = data["negative_sampler"]

    train_data = flatten_snapshots(data["train_list"], include_raw=False)
    val_data = flatten_snapshots(data["val_list"], include_raw=True)
    test_data = flatten_snapshots(data["test_list"], include_raw=True)
    trainval_data = np.vstack((train_data, val_data[:, :4])) if len(val_data) else train_data.copy()
    all_data = np.vstack((trainval_data, test_data[:, :4])) if len(test_data) else trainval_data.copy()
    num_nodes = int(data["num_nodes"])
    num_rels = int(data["num_rels"])

    rules_dict, rule_path, train_time = learn_or_load_rules(args, train_data, num_rels, out_dir)
    print(f"[TLogic-Fair] store train edges: start events={len(train_data)}", flush=True)
    t_edges = time.perf_counter()
    learn_edges = store_edges(train_data)
    store_edges_time = time.perf_counter() - t_edges
    train_time += store_edges_time
    print(
        f"[TLogic-Fair] store train edges: done relations={len(learn_edges)} "
        f"elapsed={store_edges_time:.1f}s",
        flush=True,
    )

    print(
        f"[TLogic-Fair] rules={sum(len(v) for v in rules_dict.values())} "
        f"relations_with_rules={sum(1 for v in rules_dict.values() if v)} "
        f"train_time={train_time:.3f}s",
        flush=True,
    )

    val_metrics, val_profile = evaluate_events(
        args,
        val_data,
        rules_dict,
        learn_edges,
        all_data,
        num_nodes,
        "val",
        measure_forward=False,
    )
    test_metrics, test_profile = evaluate_events(
        args,
        test_data,
        rules_dict,
        learn_edges,
        all_data,
        num_nodes,
        "test",
        measure_forward=True,
    )

    metrics = {
        "format": "tlogic_fair_v1",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "ns_q": int(args.ns_q),
        "ns_seed": int(args.ns_seed),
        "train_predict_ratio": float(args.train_predict_ratio),
        "rule_lengths": [int(x) for x in args.rule_lengths],
        "num_walks": int(args.num_walks),
        "transition_distr": args.transition_distr,
        "window": int(args.window),
        "top_k": int(args.top_k),
        "score_lambda": float(args.score_lambda),
        "score_alpha": float(args.score_alpha),
        "min_conf": float(args.min_conf),
        "min_body_supp": int(args.min_body_supp),
        "learn_rules_flag": bool(args.learn_rules_flag),
        "rule_path": rule_path,
        "num_rules_after_filter": int(sum(len(v) for v in rules_dict.values())),
        "train_time_s": float(train_time),
        "train_peak_allocated_bytes": None,
        "eval_peak_allocated_bytes": None,
        "test_forward_time_s": float(test_profile["forward_time_s"]),
        "test_inference_sample_count": int(test_profile["sample_count"]),
        "val_forward_time_s": float(val_profile["forward_time_s"]),
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
        "model_note": (
            "TGB2 TLogic temporal walks, rule learning, rule filtering, rule application, "
            "score_12, and noisy-or candidate aggregation are reused. Final ranking scores "
            "one positive plus the protocol negatives and computes EAGLE strict metrics."
        ),
    }
    save_metrics(out_dir, metrics)
    print(
        f"[TLogic-Fair] val_mrr={metrics['val_mrr']:.6f} test_mrr={metrics['test_mrr']:.6f} "
        f"test_hit1={metrics['test_hit1']:.6f} test_hit10={metrics['test_hit10']:.6f}",
        flush=True,
    )
    print(
        f"[TLogic-Fair] train_time={train_time:.3f}s "
        f"test_forward_time={test_profile['forward_time_s']:.3f}s "
        f"test_wall_time={test_profile.get('wall_time_s', 0.0):.3f}s "
        f"test_samples={test_profile['sample_count']} saved -> {out_dir}",
        flush=True,
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser("Fair TLogic trainer/evaluator for EAGLE TKG protocols.")
    parser.add_argument("--dataset", type=str, required=True, choices=eagle_utils.SUPPORTED_DATASETS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--eval_batch_size", type=int, default=200)

    parser.add_argument("--rule_lengths", "-l", type=int, nargs="+", default=[1])
    parser.add_argument("--num_walks", "-n", type=int, default=100)
    parser.add_argument("--transition_distr", type=str, default="exp", choices=("exp", "unif"))
    parser.add_argument("--window", "-w", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_processes", "-p", type=int, default=1)
    parser.add_argument("--score-lambda", dest="score_lambda", type=float, default=0.1)
    parser.add_argument("--score-alpha", dest="score_alpha", type=float, default=0.5)
    parser.add_argument("--min-conf", type=float, default=0.01)
    parser.add_argument("--min-body-supp", type=int, default=2)
    parser.add_argument("--learn-rules-flag", dest="learn_rules_flag", action="store_true", default=True)
    parser.add_argument("--load-rules", dest="learn_rules_flag", action="store_false")
    parser.add_argument("--rule-filename", type=str, default="")

    args = parser.parse_args()
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if int(args.eval_batch_size) <= 0:
        raise ValueError("--eval_batch_size must be positive")
    if int(args.num_walks) <= 0:
        raise ValueError("--num_walks must be positive")
    if int(args.num_processes) <= 0:
        raise ValueError("--num_processes must be positive")
    if int(args.top_k) <= 0:
        raise ValueError("--top_k must be positive")
    if not args.rule_lengths or any(int(x) <= 0 for x in args.rule_lengths):
        raise ValueError("--rule_lengths must contain positive integers")
    if not args.learn_rules_flag and not args.rule_filename:
        raise ValueError("--load-rules requires --rule-filename")
    return args


if __name__ == "__main__":
    run(parse_args())
