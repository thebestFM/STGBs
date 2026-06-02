import argparse
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
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    if TGB2_DIR not in sys.path:
        sys.path.insert(0, TGB2_DIR)


def import_regcn():
    ensure_tgb2_import_path()
    import torch
    from TGB2.modules.rrgcn import RecurrentRGCNREGCN
    from TGB2.modules.tkg_utils_dgl import build_sub_graph

    return torch, RecurrentRGCNREGCN, build_sub_graph


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


def make_triple_snapshots(snapshot_list):
    return [np.ascontiguousarray(events[:, :3], dtype=np.int64) for events, _, _ in snapshot_list]


def snapshot_raw_times(snapshot_list):
    return [int(t_orig) for _, _, t_orig in snapshot_list]


def make_out_dir(args):
    name = (
        f"nsq{args.ns_q}_ns{args.ns_seed}_tpr{args.train_predict_ratio:g}"
        f"_h{args.n_hidden}_ly{args.n_layers}_his{args.train_history_len}"
        f"_testhis{args.test_history_len}_lr{args.lr:g}"
    )
    return osp.join("results_regcn_fair", args.dataset, f"seed{args.seed}", name)


def build_model(args, data, torch, RecurrentRGCNREGCN):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and int(args.gpu) >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    use_cuda = device.type == "cuda"
    if not use_cuda and bool(args.self_loop):
        print(
            "[REGCN-Fair] CPU run detected. TGB2 UnionRGCNLayer hardcodes cuda() "
            "inside self-loop masking, so self_loop is disabled for CPU compatibility.",
            flush=True,
        )
        args.self_loop = False

    model = RecurrentRGCNREGCN(
        args.decoder,
        args.encoder,
        int(data["num_nodes"]),
        int(data["num_rels_raw"]),
        0,
        0,
        args.n_hidden,
        args.opn,
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
        use_static=False,
        entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction,
        use_cuda=use_cuda,
        gpu=int(args.gpu) if use_cuda else torch.device("cpu"),
        analysis=args.run_analysis,
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
    losses_e = []
    losses_r = []
    losses_static = []
    start_time = time.perf_counter()
    for snap_idx in order:
        if snap_idx == 0 or len(train_snaps[snap_idx]) == 0:
            continue
        local_start = max(0, snap_idx - int(args.train_history_len))
        history_glist = build_history_graphs(
            train_snaps, local_start, snap_idx, num_nodes, num_rels, use_cuda, args, build_sub_graph
        )
        if not history_glist:
            continue
        output = torch.from_numpy(train_snaps[snap_idx]).long().to(device)
        loss_e, loss_r, loss_static = model.get_loss(history_glist, output, None, use_cuda)
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


def regcn_candidate_scores(model, history_glist, batch_tensor, neg_arr, neg_mask, batch_size_total, device, torch):
    evolve_embs, _, r_emb, _, _ = model.forward(history_glist, None, device.type == "cuda")
    embedding = torch.nn.functional.normalize(evolve_embs[-1]) if model.layer_norm else evolve_embs[-1]

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

        score = model.decoder_ob.forward(
            embedding,
            r_emb,
            batch_tensor[query_id].unsqueeze(0),
            samples_of_interest_emb=embedding[candidates],
            batch_size_total=int(batch_size_total),
        ).squeeze(0)
        scores_np = score.detach().cpu().numpy().astype(np.float32, copy=False)
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
                neg_clean = neg_arr.copy()
                neg_clean[neg_clean < 0] = 0
                batch_tensor = torch.from_numpy(batch.astype(np.int64, copy=False)).long().to(device)
                if measure_forward:
                    sync_device(torch, device)
                    t0 = time.perf_counter()
                pos_scores, neg_scores = regcn_candidate_scores(
                    model, history_glist, batch_tensor, neg_clean, neg_mask, len(snap), device, torch
                )
                if measure_forward:
                    sync_device(torch, device)
                    forward_time += time.perf_counter() - t0
                add_metric_sums(sums, compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask))
                sample_count += int(len(batch))

    metrics = finalize_metric_sums(sums)
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    return metrics, {"forward_time_s": float(forward_time), "sample_count": int(sample_count)}


def run(args):
    torch, RecurrentRGCNREGCN, build_sub_graph = import_regcn()
    set_random_seed(args.seed)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[REGCN-Fair] disabled cuDNN backend for REGCN ConvTransE stability.", flush=True)

    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[REGCN-Fair]")
    if data.get("is_thg"):
        raise ValueError("REGCN-Fair is adapted for TKG datasets only; Yelp-* THG datasets are unsupported here.")

    out_dir = make_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    save_config(out_dir, {k: v for k, v in vars(args).items() if not k.startswith("_")})

    train_snaps = make_triple_snapshots(data["train_list"])
    val_snaps = make_triple_snapshots(data["val_list"])
    test_snaps = make_triple_snapshots(data["test_list"])
    val_raw_times = snapshot_raw_times(data["val_list"])
    test_raw_times = snapshot_raw_times(data["test_list"])
    all_snaps = train_snaps + val_snaps + test_snaps
    val_start = len(train_snaps)
    test_start = len(train_snaps) + len(val_snaps)
    args._negative_sampler = data["negative_sampler"]

    model, device, use_cuda = build_model(args, data, torch, RecurrentRGCNREGCN)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(args, optimizer, torch)
    num_nodes = int(data["num_nodes"])
    num_rels = int(data["num_rels"])

    print(
        f"[REGCN-Fair] model nodes={num_nodes} rels={num_rels} raw_rels={data['num_rels_raw']} "
        f"device={device} train_history={args.train_history_len} test_history={args.test_history_len}",
        flush=True,
    )

    checkpoint_path = osp.join(out_dir, "best_model.pt")
    best_val = -float("inf")
    best_epoch = 0
    train_time_total = 0.0
    early_stopped = False
    early_stop_epoch = 0
    epoch_logs = []

    reset_cuda_peak(torch, device)
    for epoch in range(int(args.n_epochs)):
        log = train_one_epoch(
            args,
            epoch,
            model,
            optimizer,
            scheduler,
            train_snaps,
            device,
            use_cuda,
            num_nodes,
            num_rels,
            torch,
            build_sub_graph,
        )
        train_time_total += log["train_time_s"]
        do_val = epoch and epoch % int(args.evaluate_every) == 0
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
                int(args.test_history_len),
                torch,
                build_sub_graph,
                measure_forward=False,
            )
            log["val_mrr_strict"] = float(val_metrics["mrr_strict"])
            log["val_hit@1_strict"] = float(val_metrics["hit@1_strict"])
            log["val_hit@10_strict"] = float(val_metrics["hit@10_strict"])
            if val_metrics["mrr_strict"] >= best_val + float(args.tolerance):
                best_val = float(val_metrics["mrr_strict"])
                best_epoch = int(epoch)
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, checkpoint_path)
            log["epochs_since_best"] = int(epoch - best_epoch) if best_epoch else 0
        epoch_logs.append(log)
        print(
            f"[REGCN-Fair] epoch={epoch} loss={log['loss']:.5f} "
            f"ent={log['loss_entity']:.5f} rel={log['loss_relation']:.5f} "
            f"static={log['loss_static']:.5f} train_time={log['train_time_s']:.2f}s "
            f"best_val_mrr={max(best_val, 0.0):.5f}",
            flush=True,
        )
        if do_val and best_epoch and int(epoch - best_epoch) >= int(args.patience):
            early_stopped = True
            early_stop_epoch = int(epoch)
            print(
                f"[REGCN-Fair] early stop at epoch={epoch}: val_mrr did not improve for "
                f"{epoch - best_epoch} epochs (patience={args.patience}, best_epoch={best_epoch})",
                flush=True,
            )
            break

    train_peak = cuda_peak_allocated(torch, device)
    if osp.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        best_epoch = int(args.n_epochs) - 1
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
            int(args.test_history_len),
            torch,
            build_sub_graph,
            measure_forward=False,
        )
        best_val = float(val_metrics["mrr_strict"])
        torch.save({"state_dict": model.state_dict(), "epoch": best_epoch, "val_metrics": val_metrics}, checkpoint_path)

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
        int(args.test_history_len),
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
        int(args.test_history_len),
        torch,
        build_sub_graph,
        measure_forward=True,
    )
    eval_peak = cuda_peak_allocated(torch, device)

    metrics = {
        "format": "regcn_fair_v1",
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
            "TGB2 RecurrentRGCNREGCN/get_loss/build_sub_graph are used. Training keeps "
            "REGCN's full-entity/full-relation supervised losses; final val/test ranking "
            "scores exactly one positive plus the protocol negatives with EAGLE strict metrics."
        ),
    }
    save_metrics(out_dir, metrics)
    print(
        f"[REGCN-Fair] final val_mrr={metrics['val_mrr']:.6f} test_mrr={metrics['test_mrr']:.6f} "
        f"test_hit1={metrics['test_hit1']:.6f} test_hit10={metrics['test_hit10']:.6f}",
        flush=True,
    )
    print(
        f"[REGCN-Fair] train_time={train_time_total:.3f}s "
        f"test_forward_time={test_profile['forward_time_s']:.3f}s "
        f"test_samples={test_profile['sample_count']} "
        f"train_peak={format_bytes(train_peak)} eval_peak={format_bytes(eval_peak)} saved -> {out_dir}",
        flush=True,
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser("Fair REGCN trainer for EAGLE TKG protocols.")
    parser.add_argument("--dataset", type=str, required=True, choices=eagle_utils.SUPPORTED_DATASETS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--disable-cudnn", action="store_true", default=True)
    parser.add_argument("--enable-cudnn", dest="disable_cudnn", action="store_false")

    parser.add_argument("--run-analysis", action="store_true", default=False)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--task-weight", type=float, default=0.7)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--angle", type=int, default=10)
    parser.add_argument("--encoder", type=str, default="uvrgcn")
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
    parser.add_argument("--layer-norm", action="store_true", default=True)
    parser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false")
    parser.add_argument("--relation-prediction", action="store_true", default=False)
    parser.add_argument("--entity-prediction", action="store_true", default=True)
    parser.add_argument("--no-entity-prediction", dest="entity_prediction", action="store_false")

    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--lr-scheduler", type=str, default="none", choices=("none", "step"))
    parser.add_argument("--lr-step-size", type=int, default=50)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-step", type=str, default="epoch", choices=("epoch", "batch"))

    parser.add_argument("--decoder", type=str, default="convtranse")
    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dropout", type=float, default=0.2)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--train-history-len", type=int, default=3)
    parser.add_argument("--test-history-len", type=int, default=3)
    parser.add_argument("--shuffle-snapshots", action="store_true", default=True)
    parser.add_argument("--no-shuffle-snapshots", dest="shuffle_snapshots", action="store_false")

    args = parser.parse_args()
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if int(args.eval_batch_size) <= 0:
        raise ValueError("--eval_batch_size must be positive")
    if int(args.n_epochs) <= 0:
        raise ValueError("--n-epochs must be positive")
    if int(args.evaluate_every) <= 0:
        raise ValueError("--evaluate-every must be positive")
    if int(args.patience) <= 0:
        raise ValueError("--patience must be positive")
    if int(args.train_history_len) <= 0 or int(args.test_history_len) <= 0:
        raise ValueError("--train-history-len and --test-history-len must be positive")
    if not bool(args.entity_prediction) and not bool(args.relation_prediction):
        raise ValueError("REGCN training needs at least one of entity/relation prediction enabled")
    if args.encoder != "uvrgcn":
        raise ValueError("TGB2 REGCN currently supports --encoder uvrgcn")
    if args.decoder != "convtranse":
        raise ValueError("TGB2 REGCN currently supports --decoder convtranse")
    return args


if __name__ == "__main__":
    run(parse_args())
