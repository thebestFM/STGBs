import argparse
import json
import os
import os.path as osp
import time
from types import SimpleNamespace

import optuna

import train_tirgn_fair
import utils as eagle_utils


def sample_trial_args(cli, trial):
    hidden = trial.suggest_categorical("n_hidden", cli.hidden_choices)
    n_layers = trial.suggest_categorical("n_layers", [1, 2])
    train_history_len = trial.suggest_categorical("train_history_len", cli.history_choices)
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
    decoder_dropout = trial.suggest_float("decoder_dropout", 0.0, 0.4, step=0.1)
    n_bases = trial.suggest_categorical("n_bases", cli.n_bases_candidates)
    if int(n_bases) > int(hidden) or int(hidden) % int(n_bases) != 0:
        raise optuna.TrialPruned(f"invalid n_hidden/n_bases pair: {hidden}/{n_bases}")
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)

    entity_prediction = not cli.relation_only
    relation_prediction = True
    task_weight = 0.0 if cli.relation_only else trial.suggest_float("task_weight", 0.5, 0.9, step=0.1)

    return SimpleNamespace(
        dataset=cli.dataset,
        seed=cli.seed,
        ns_q=cli.ns_q,
        ns_seed=cli.ns_seed,
        train_predict_ratio=cli.train_predict_ratio,
        gpu=cli.gpu,
        batch_size=1,
        eval_batch_size=cli.eval_batch_size,
        test=False,
        run_analysis=False,
        run_statistic=False,
        multi_step=False,
        topk=50,
        add_static_graph=cli.add_static_graph,
        add_rel_word=False,
        relation_evaluation=False,
        weight=trial.suggest_float("weight", 0.3, 0.7, step=0.1),
        task_weight=task_weight,
        discount=1.0,
        angle=trial.suggest_categorical("angle", [8, 10, 12, 14]),
        encoder="convgcn",
        aggregation="none",
        dropout=dropout,
        skip_connect=trial.suggest_categorical("skip_connect", [False, True]),
        n_hidden=hidden,
        opn="sub",
        n_bases=n_bases,
        n_basis=n_bases,
        n_layers=n_layers,
        self_loop=True,
        layer_norm=trial.suggest_categorical("layer_norm", [False, True]),
        relation_prediction=relation_prediction,
        entity_prediction=entity_prediction,
        split_by_relation=False,
        n_epochs=cli.n_epochs,
        lr=lr,
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        grad_norm=1.0,
        evaluate_every=cli.evaluate_every,
        patience=cli.patience,
        tolerance=1e-8,
        lr_scheduler="none",
        lr_step_size=50,
        lr_gamma=0.5,
        scheduler_step="epoch",
        decoder="timeconvtranse",
        input_dropout=decoder_dropout,
        hidden_dropout=decoder_dropout,
        feat_dropout=decoder_dropout,
        train_history_len=train_history_len,
        test_history_len=train_history_len,
        dilate_len=1,
        grid_search=False,
        tune="optuna",
        num_k=500,
        history_rate=trial.suggest_float("history_rate", 0.1, 0.7, step=0.1),
        save=f"optuna_t{trial.number:03d}",
        shuffle_snapshots=True,
    )


def clear_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def make_objective(cli):
    def objective(trial):
        args = sample_trial_args(cli, trial)
        try:
            metrics = train_tirgn_fair.run(args)
        except RuntimeError as exc:
            clear_cuda_cache()
            message = str(exc).lower()
            if "out of memory" in message or "cuda error" in message:
                raise optuna.TrialPruned(f"trial pruned after CUDA failure: {exc}") from exc
            raise
        finally:
            clear_cuda_cache()
        value = float(metrics["val_mrr"])
        trial.set_user_attr("best_epoch", int(metrics["best_epoch"]))
        trial.set_user_attr("test_mrr", float(metrics["test_mrr"]))
        trial.set_user_attr("test_hit1", float(metrics["test_hit1"]))
        trial.set_user_attr("test_hit10", float(metrics["test_hit10"]))
        trial.set_user_attr("train_time_s", float(metrics["train_time_s"]))
        return value

    return objective


def parse_args():
    parser = argparse.ArgumentParser("Optuna tuner for train_tirgn_fair.py")
    parser.add_argument("--dataset", type=str, required=True, choices=eagle_utils.SUPPORTED_DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=9999)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--study-name", type=str, default="")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--add-static-graph", action="store_true", default=False)
    parser.add_argument("--relation-only", action="store_true", default=False)
    parser.add_argument(
        "--hidden-choices",
        type=int,
        nargs="+",
        default=[32, 48, 64, 96, 128, 200],
        help="Candidate hidden sizes. Keep this small for tkgl-polecat/Yelp to avoid OOM.",
    )
    parser.add_argument(
        "--history-choices",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 12],
        help="Candidate train/test history lengths.",
    )
    parser.add_argument(
        "--n-bases-candidates",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--output-dir", type=str, default="results_tirgn_optuna")
    args = parser.parse_args()
    if int(args.n_trials) <= 0:
        raise ValueError("--n-trials must be positive")
    if int(args.n_epochs) <= 0:
        raise ValueError("--n-epochs must be positive")
    if int(args.evaluate_every) <= 0:
        raise ValueError("--evaluate-every must be positive")
    if int(args.patience) <= 0:
        raise ValueError("--patience must be positive")
    return args


def main():
    cli = parse_args()
    study_name = cli.study_name or f"tirgn_fair_{cli.dataset}_seed{cli.seed}_{int(time.time())}"
    sampler = optuna.samplers.TPESampler(seed=cli.seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=cli.storage or None,
        load_if_exists=bool(cli.storage),
    )
    study.optimize(make_objective(cli), n_trials=cli.n_trials)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("all Optuna trials were pruned or failed; reduce the search space or model size")

    out_dir = osp.join(cli.output_dir, cli.dataset, f"seed{cli.seed}")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = osp.join(out_dir, f"{study.study_name}.json")
    summary = {
        "study_name": study.study_name,
        "dataset": cli.dataset,
        "seed": cli.seed,
        "ns_q": cli.ns_q,
        "ns_seed": cli.ns_seed,
        "n_trials": cli.n_trials,
        "best_value_val_mrr": study.best_value,
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
        "search_space_note": (
            "Searches hidden size, layers, history length/rate, lr, weight decay, "
            "dropouts, n_bases, angle, skip connection, layer norm, and TiRGN loss weights. "
            "The objective is strict validation MRR from train_tirgn_fair.py."
        ),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[TiRGN-Optuna] best val_mrr={study.best_value:.6f}", flush=True)
    print(f"[TiRGN-Optuna] best params={study.best_trial.params}", flush=True)
    print(f"[TiRGN-Optuna] saved -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
