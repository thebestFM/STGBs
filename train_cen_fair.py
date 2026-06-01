import argparse
import json
import os
import os.path as osp
import random
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
    """Make both TGB2.* and the original modules.* imports resolvable."""
    if not osp.isdir(osp.join(TGB2_DIR, "modules")):
        raise FileNotFoundError(f"TGB2 modules directory not found under {TGB2_DIR}")
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    if TGB2_DIR not in sys.path:
        sys.path.insert(0, TGB2_DIR)


def import_cen():
    ensure_tgb2_import_path()
    import torch
    from TGB2.modules.rrgcn import RecurrentRGCNCEN
    from TGB2.modules.tkg_utils_dgl import build_sub_graph

    return torch, RecurrentRGCNCEN, build_sub_graph


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


def snapshot_triples(snapshot):
    events, _, _ = snapshot
    return np.ascontiguousarray(events[:, :3], dtype=np.int64)


def make_triple_snapshots(snapshot_list):
    return [snapshot_triples(s) for s in snapshot_list]


def snapshot_raw_times(snapshot_list):
    return [int(t_orig) for _, _, t_orig in snapshot_list]


def make_out_dir(args):
    name = (
        f"nsq{args.ns_q}_ns{args.ns_seed}_tpr{args.train_predict_ratio:g}"
        f"_h{args.n_hidden}_ly{args.n_layers}_start{args.start_history_len}_his{args.train_history_len}"
        f"_testhis{args.test_history_len}_lr{args.lr:g}"
    )
    return osp.join("results_cen_fair", args.dataset, f"seed{args.seed}", name)


def build_model(args, data, torch, RecurrentRGCNCEN):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and int(args.gpu) >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    use_cuda = device.type == "cuda"
    if not use_cuda and bool(args.self_loop):
        print(
            "[CEN-Fair] CPU run detected. TGB2 CEN's UnionRGCNLayer hardcodes cuda() "
            "inside self-loop masking, so self_loop is disabled for CPU compatibility.",
            flush=True,
        )
        args.self_loop = False

    sequence_len = int(args.train_history_len)
    gpu_arg = int(args.gpu) if use_cuda else torch.device("cpu")
    model = RecurrentRGCNCEN(
        args.decoder,
        args.encoder,
        int(data["num_nodes"]),
        int(data["num_rels"]),
        args.n_hidden,
        args.opn,
        sequence_len=sequence_len,
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
        entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction,
        use_cuda=use_cuda,
        gpu=gpu_arg,
    )
    model.to(device)
    return model, device, use_cuda


def build_history_graphs(triple_snapshots, start, end, num_nodes, num_rels, use_cuda, args, build_sub_graph):
    start = max(0, int(start))
    end = max(start, int(end))
    return [
        build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu)
        for snap in triple_snapshots[start:end]
        if len(snap)
    ]


def make_scheduler(args, optimizer, torch):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma)
        )
    raise ValueError(f"unsupported lr scheduler: {args.lr_scheduler}")


def train_one_epoch(
    args,
    epoch,
    model,
    optimizer,
    scheduler,
    train_snaps,
    history_len,
    device,
    use_cuda,
    num_nodes,
    num_rels,
    torch,
    build_sub_graph,
):
    model.train()
    order = list(range(len(train_snaps)))
    if args.shuffle_snapshots:
        random.shuffle(order)

    losses = []
    start_time = time.perf_counter()
    for snap_idx in order:
        if snap_idx == 0 or snap_idx == 1 or len(train_snaps[snap_idx]) == 0:
            continue
        local_start = max(0, snap_idx - int(history_len))
        history_glist = build_history_graphs(
            train_snaps, local_start, snap_idx, num_nodes, num_rels, use_cuda, args, build_sub_graph
        )
        if not history_glist:
            continue
        output = torch.from_numpy(train_snaps[snap_idx]).long().to(device)
        loss = model.get_loss(history_glist, output, None, use_cuda)
        losses.append(float(loss.detach().cpu().item()))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None and args.scheduler_step == "batch":
            scheduler.step()

    if scheduler is not None and args.scheduler_step == "epoch":
        scheduler.step()
    sync_device(torch, device)
    elapsed = time.perf_counter() - start_time
    return {
        "epoch": int(epoch),
        "history_len": int(history_len),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "train_time_s": float(elapsed),
    }


def cen_candidate_scores(model, history_glist, batch_tensor, neg_arr, neg_mask, batch_size_total, device, torch):
    """Score exactly one positive plus the protocol negatives for each query."""
    evolve_embeddings = []
    for idx in range(len(history_glist)):
        evolve_embs, r_emb = model.forward(history_glist[idx:], device.type == "cuda")
        evolve_embeddings.append(evolve_embs[-1])
    evolve_embeddings.reverse()

    batch_query_count = int(batch_tensor.shape[0])
    pos_scores = np.zeros((batch_query_count, 1), dtype=np.float32)
    neg_scores = np.zeros_like(neg_arr, dtype=np.float32)

    for query_id in range(batch_query_count):
        valid_len = int(np.sum(neg_mask[query_id]))
        pos = batch_tensor[query_id, 2].view(1)
        if valid_len:
            neg = torch.from_numpy(neg_arr[query_id, :valid_len].astype(np.int64, copy=False)).long().to(device)
            candidates = torch.cat((pos, neg), dim=0)
        else:
            candidates = pos

        samples = [emb[candidates] for emb in evolve_embeddings]
        score_list = model.decoder_ob.forward(
            evolve_embeddings,
            r_emb,
            batch_tensor[query_id].unsqueeze(0),
            samples_of_interest_emb=samples,
            batch_size_total=int(batch_size_total),
        )
        score_list = [score.unsqueeze(2) for score in score_list]
        scores = torch.cat(score_list, dim=2)
        scores = torch.softmax(scores, dim=1)
        scores = torch.sum(scores, dim=-1).squeeze(0)

        scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)
        pos_scores[query_id, 0] = scores_np[0]
        if valid_len:
            neg_scores[query_id, :valid_len] = scores_np[1:]

    return pos_scores, neg_scores


def evaluate_split(
    args,
    model,
    split_name,
    eval_snaps,
    eval_raw_times,
    all_snaps,
    device,
    use_cuda,
    num_nodes,
    num_rels,
    split_start_idx,
    history_len,
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
        for local_idx, snap in enumerate(eval_snaps):
            if len(snap) == 0:
                continue
            global_idx = int(split_start_idx + local_idx)
            local_start = max(0, global_idx - int(history_len))
            history_glist = build_history_graphs(
                all_snaps, local_start, global_idx, num_nodes, num_rels, use_cuda, args, build_sub_graph
            )
            if not history_glist:
                continue
            raw_t = int(eval_raw_times[local_idx])
            for batch, neg_arr, neg_mask in collect_eval_batch(
                snap[:, :3], raw_t, neg_sampler, split_name, args.eval_batch_size
            ):
                if len(batch) == 0:
                    continue
                neg_clean = neg_arr.copy()
                neg_clean[neg_clean < 0] = 0
                batch_tensor = torch.from_numpy(batch.astype(np.int64, copy=False)).long().to(device)

                if measure_forward:
                    sync_device(torch, device)
                    t0 = time.perf_counter()
                pos_scores, neg_scores = cen_candidate_scores(
                    model, history_glist, batch_tensor, neg_clean, neg_mask, len(snap), device, torch
                )
                if measure_forward:
                    sync_device(torch, device)
                    forward_time += time.perf_counter() - t0

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


def run_history_stage(
    args,
    model,
    stage_name,
    history_len,
    lr,
    patience,
    checkpoint_path,
    train_snaps,
    val_snaps,
    val_raw_times,
    all_snaps,
    device,
    use_cuda,
    num_nodes,
    num_rels,
    val_start,
    torch,
    build_sub_graph,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=args.weight_decay)
    scheduler = make_scheduler(args, optimizer, torch)
    best_val = -float("inf")
    best_epoch = 0
    early_stopped = False
    early_stop_epoch = 0
    train_time_total = 0.0
    logs = []

    for epoch in range(int(args.n_epochs)):
        log = train_one_epoch(
            args,
            epoch,
            model,
            optimizer,
            scheduler,
            train_snaps,
            history_len,
            device,
            use_cuda,
            num_nodes,
            num_rels,
            torch,
            build_sub_graph,
        )
        log["stage"] = stage_name
        log["lr"] = float(lr)
        train_time_total += log["train_time_s"]

        do_val = epoch % int(args.evaluate_every) == 0
        improved = False
        if do_val:
            val_metrics, _ = evaluate_split(
                args,
                model,
                "val",
                val_snaps,
                val_raw_times,
                all_snaps,
                device,
                use_cuda,
                num_nodes,
                num_rels,
                val_start,
                history_len,
                torch,
                build_sub_graph,
                measure_forward=False,
            )
            log["val_mrr_strict"] = float(val_metrics["mrr_strict"])
            log["val_hit@1_strict"] = float(val_metrics["hit@1_strict"])
            log["val_hit@10_strict"] = float(val_metrics["hit@10_strict"])
            improved = val_metrics["mrr_strict"] >= best_val
            if improved:
                best_val = float(val_metrics["mrr_strict"])
                best_epoch = int(epoch)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "history_len": int(history_len),
                        "stage": stage_name,
                        "val_metrics": val_metrics,
                    },
                    checkpoint_path,
                )
            log["epochs_since_best"] = int(epoch - best_epoch)

        logs.append(log)
        print(
            f"[CEN-Fair] stage={stage_name} history={history_len} epoch={epoch} "
            f"loss={log['loss']:.5f} train_time={log['train_time_s']:.2f}s "
            f"best_val_mrr={max(best_val, 0.0):.5f}",
            flush=True,
        )

        if do_val and (not improved) and int(epoch - best_epoch) > int(patience):
            early_stopped = True
            early_stop_epoch = int(epoch)
            print(
                f"[CEN-Fair] early stop stage={stage_name} history={history_len} at epoch={epoch}: "
                f"val_mrr did not improve for {epoch - best_epoch} epochs "
                f"(patience={patience}, best_epoch={best_epoch})",
                flush=True,
            )
            break

    if not osp.exists(checkpoint_path):
        best_epoch = int(args.n_epochs)
        val_metrics, _ = evaluate_split(
            args,
            model,
            "val",
            val_snaps,
            val_raw_times,
            all_snaps,
            device,
            use_cuda,
            num_nodes,
            num_rels,
            val_start,
            history_len,
            torch,
            build_sub_graph,
            measure_forward=False,
        )
        best_val = float(val_metrics["mrr_strict"])
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": best_epoch,
                "history_len": int(history_len),
                "stage": stage_name,
                "val_metrics": val_metrics,
            },
            checkpoint_path,
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return {
        "stage": stage_name,
        "history_len": int(history_len),
        "checkpoint_path": checkpoint_path,
        "best_mrr": float(best_val),
        "best_epoch": int(best_epoch),
        "early_stopped": bool(early_stopped),
        "early_stop_epoch": int(early_stop_epoch),
        "patience": int(patience),
        "train_time_s": float(train_time_total),
        "logs": logs,
    }


def run(args):
    torch, RecurrentRGCNCEN, build_sub_graph = import_cen()
    set_random_seed(args.seed)

    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[CEN-Fair]")
    if data.get("is_thg"):
        raise ValueError(
            "CEN curriculum training is currently adapted for TKG/TGB-style datasets only; "
            "Yelp-* THG datasets are intentionally left unsupported for this trainer."
        )
    out_dir = make_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    train_snaps = make_triple_snapshots(data["train_list"])
    val_snaps = make_triple_snapshots(data["val_list"])
    test_snaps = make_triple_snapshots(data["test_list"])
    val_raw_times = snapshot_raw_times(data["val_list"])
    test_raw_times = snapshot_raw_times(data["test_list"])
    all_snaps = train_snaps + val_snaps + test_snaps
    val_start = len(train_snaps)
    test_start = len(train_snaps) + len(val_snaps)
    args._negative_sampler = data["negative_sampler"]

    model, device, use_cuda = build_model(args, data, torch, RecurrentRGCNCEN)
    save_config(out_dir, {k: v for k, v in vars(args).items() if not k.startswith("_")})
    num_nodes = int(data["num_nodes"])
    num_rels = int(data["num_rels"])

    print(
        f"[CEN-Fair] model nodes={num_nodes} rels={num_rels} raw_rels={data['num_rels_raw']} "
        f"device={device} start_history={args.start_history_len} max_train_history={args.train_history_len}",
        flush=True,
    )

    checkpoint_path = osp.join(out_dir, "best_model.pt")
    train_time_total = 0.0
    epoch_logs = []
    history_stage_logs = []
    curriculum_stopped = False
    curriculum_stop_history_len = 0
    selected_history_len = int(args.start_history_len)
    selected_checkpoint_path = osp.join(out_dir, f"history_len_{selected_history_len}.pt")
    selected_stage = None
    best_history_mrr = -float("inf")

    reset_cuda_peak(torch, device)
    pretrain_stage = run_history_stage(
        args,
        model,
        "pretrain",
        int(args.start_history_len),
        float(args.lr),
        int(args.patience),
        selected_checkpoint_path,
        train_snaps,
        val_snaps,
        val_raw_times,
        all_snaps,
        device,
        use_cuda,
        num_nodes,
        num_rels,
        val_start,
        torch,
        build_sub_graph,
    )
    train_time_total += pretrain_stage["train_time_s"]
    epoch_logs.extend(pretrain_stage["logs"])
    history_stage_logs.append({k: v for k, v in pretrain_stage.items() if k != "logs"})
    selected_stage = pretrain_stage
    best_history_mrr = float(pretrain_stage["best_mrr"])

    for history_len in range(int(args.start_history_len) + 1, int(args.train_history_len) + 1):
        stage_checkpoint_path = osp.join(out_dir, f"history_len_{history_len}.pt")
        stage = run_history_stage(
            args,
            model,
            f"curriculum_h{history_len}",
            history_len,
            0.1 * float(args.lr),
            int(args.curriculum_patience),
            stage_checkpoint_path,
            train_snaps,
            val_snaps,
            val_raw_times,
            all_snaps,
            device,
            use_cuda,
            num_nodes,
            num_rels,
            val_start,
            torch,
            build_sub_graph,
        )
        train_time_total += stage["train_time_s"]
        epoch_logs.extend(stage["logs"])
        history_stage_logs.append({k: v for k, v in stage.items() if k != "logs"})

        if float(stage["best_mrr"]) < best_history_mrr:
            curriculum_stopped = True
            curriculum_stop_history_len = int(history_len)
            print(
                f"[CEN-Fair] curriculum stop: history={history_len} val_mrr={stage['best_mrr']:.6f} "
                f"< best_history_mrr={best_history_mrr:.6f}; selected_history={selected_history_len}",
                flush=True,
            )
            break

        selected_history_len = int(history_len)
        selected_checkpoint_path = stage_checkpoint_path
        selected_stage = stage
        best_history_mrr = max(best_history_mrr, float(stage["best_mrr"]))

    train_peak = cuda_peak_allocated(torch, device)
    ckpt = torch.load(selected_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": int(ckpt.get("epoch", 0)),
            "history_len": int(selected_history_len),
            "stage": ckpt.get("stage", ""),
            "val_metrics": ckpt.get("val_metrics", {}),
        },
        checkpoint_path,
    )
    args.test_history_len = int(selected_history_len)

    val_metrics, _ = evaluate_split(
        args,
        model,
        "val",
        val_snaps,
        val_raw_times,
        all_snaps,
        device,
        use_cuda,
        num_nodes,
        num_rels,
        val_start,
        selected_history_len,
        torch,
        build_sub_graph,
        measure_forward=False,
    )
    reset_cuda_peak(torch, device)
    test_metrics, test_profile = evaluate_split(
        args,
        model,
        "test",
        test_snaps,
        test_raw_times,
        all_snaps,
        device,
        use_cuda,
        num_nodes,
        num_rels,
        test_start,
        selected_history_len,
        torch,
        build_sub_graph,
        measure_forward=True,
    )
    eval_peak = cuda_peak_allocated(torch, device)

    metrics = {
        "format": "cen_fair_v1",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "ns_q": int(args.ns_q),
        "ns_seed": int(args.ns_seed),
        "train_predict_ratio": float(args.train_predict_ratio),
        "best_epoch": int(selected_stage["best_epoch"] if selected_stage else 0),
        "best_val_mrr": float(best_history_mrr),
        "selected_history_len": int(selected_history_len),
        "selected_checkpoint_path": selected_checkpoint_path,
        "curriculum_stopped": bool(curriculum_stopped),
        "curriculum_stop_history_len": int(curriculum_stop_history_len),
        "stage_logs": history_stage_logs,
        "selection_metric": "val_mrr_strict",
        "early_stop_metric": "val_mrr_strict",
        "early_stopped": bool(any(stage["early_stopped"] for stage in history_stage_logs)),
        "early_stop_epoch": int(selected_stage["early_stop_epoch"] if selected_stage else 0),
        "patience": int(args.patience),
        "curriculum_patience": int(args.curriculum_patience),
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
            "TGB2 RecurrentRGCNCEN/get_loss/build_sub_graph are used with CEN-style "
            "pretraining and curriculum history-length selection. The shared dataset "
            "reverse-edge policy is unchanged. Evaluation scores exactly one positive "
            "plus the protocol negatives and computes EAGLE strict metrics."
        ),
    }
    save_metrics(out_dir, metrics)
    print(
        f"[CEN-Fair] selected_history={selected_history_len} final val_mrr={metrics['val_mrr']:.6f} "
        f"test_mrr={metrics['test_mrr']:.6f} test_hit1={metrics['test_hit1']:.6f} "
        f"test_hit10={metrics['test_hit10']:.6f}",
        flush=True,
    )
    print(
        f"[CEN-Fair] train_time={train_time_total:.3f}s "
        f"test_forward_time={test_profile['forward_time_s']:.3f}s "
        f"test_samples={test_profile['sample_count']} "
        f"train_peak={format_bytes(train_peak)} eval_peak={format_bytes(eval_peak)}",
        flush=True,
    )
    print(f"[CEN-Fair] saved -> {out_dir}", flush=True)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser("Fair CEN trainer for EAGLE TKG/THG protocols.")
    parser.add_argument("--dataset", type=str, required=True, choices=eagle_utils.SUPPORTED_DATASETS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=256)

    parser.add_argument("--encoder", type=str, default="uvrgcn")
    parser.add_argument("--decoder", type=str, default="convtranse")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--skip-connect", action="store_true", default=False)
    parser.add_argument("--n-hidden", type=int, default=200)
    parser.add_argument("--opn", type=str, default="sub")
    parser.add_argument("--n-bases", type=int, default=100)
    parser.add_argument("--n-basis", type=int, default=100)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--self-loop", action="store_true", default=True)
    parser.add_argument("--no-self-loop", dest="self_loop", action="store_false")
    parser.add_argument("--layer-norm", action="store_true", default=True)
    parser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false")
    parser.add_argument("--relation-prediction", action="store_true", default=False)
    parser.add_argument("--entity-prediction", action="store_true", default=True)
    parser.add_argument("--no-entity-prediction", dest="entity_prediction", action="store_false")

    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--curriculum-patience", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--lr-scheduler", type=str, default="none", choices=("none", "step"))
    parser.add_argument("--lr-step-size", type=int, default=50)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-step", type=str, default="epoch", choices=("epoch", "batch"))

    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dropout", type=float, default=0.2)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--start-history-len", type=int, default=3)
    parser.add_argument("--train-history-len", type=int, default=3)
    parser.add_argument("--test-history-len", type=int, default=3)
    parser.add_argument("--shuffle-snapshots", action="store_true", default=True)
    parser.add_argument("--no-shuffle-snapshots", dest="shuffle_snapshots", action="store_false")

    args = parser.parse_args()
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if not 0.0 <= float(args.train_predict_ratio) <= 1.0:
        raise ValueError("--train_predict_ratio must be in [0, 1]")
    if int(args.evaluate_every) <= 0:
        raise ValueError("--evaluate-every must be positive")
    if int(args.n_epochs) <= 0:
        raise ValueError("--n-epochs must be positive")
    if int(args.patience) <= 0:
        raise ValueError("--patience must be positive")
    if int(args.curriculum_patience) <= 0:
        raise ValueError("--curriculum-patience must be positive")
    if int(args.n_hidden) <= 0:
        raise ValueError("--n-hidden must be positive")
    if int(args.n_bases) <= 0:
        raise ValueError("--n-bases must be positive")
    if int(args.start_history_len) <= 0 or int(args.train_history_len) <= 0 or int(args.test_history_len) <= 0:
        raise ValueError("--start-history-len, --train-history-len and --test-history-len must be positive")
    if int(args.start_history_len) > int(args.train_history_len):
        raise ValueError("--start-history-len must be <= --train-history-len for CEN curriculum training")
    if int(args.eval_batch_size) <= 0:
        raise ValueError("--eval_batch_size must be positive")
    if not bool(args.entity_prediction):
        raise ValueError("CEN get_loss needs --entity-prediction enabled for supervised training")
    if args.encoder != "uvrgcn":
        raise ValueError("TGB2 CEN currently supports --encoder uvrgcn")
    if args.decoder != "convtranse":
        raise ValueError("TGB2 CEN currently supports --decoder convtranse")
    return args


if __name__ == "__main__":
    run(parse_args())
