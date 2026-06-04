import argparse
import importlib.util
import os
import os.path as osp
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
TIDFORMER_DIR = osp.join(REPO_DIR, "baseline_TIDFormer")


def _load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_tidformer_official():
    """Load TIDFormer's official modules without shadowing this repo's utils.py."""
    if not osp.isdir(TIDFORMER_DIR):
        raise FileNotFoundError(f"TIDFormer directory not found: {TIDFORMER_DIR}")

    saved = {
        name: sys.modules.get(name)
        for name in ("utils", "utils.utils", "utils.DataLoader", "models", "models.modules")
    }
    missing = {name for name, value in saved.items() if value is None}
    try:
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [osp.join(TIDFORMER_DIR, "utils")]
        models_pkg = types.ModuleType("models")
        models_pkg.__path__ = [osp.join(TIDFORMER_DIR, "models")]
        sys.modules["utils"] = utils_pkg
        sys.modules["models"] = models_pkg

        tid_utils = _load_module("utils.utils", osp.join(TIDFORMER_DIR, "utils", "utils.py"))
        tid_modules = _load_module("models.modules", osp.join(TIDFORMER_DIR, "models", "modules.py"))
        sys.modules["utils.utils"] = tid_utils
        sys.modules["models.modules"] = tid_modules
        tid_model = _load_module("tidformer_official_model", osp.join(TIDFORMER_DIR, "models", "TIDFormer.py"))
    finally:
        for name, value in saved.items():
            if name in missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    tid_dataloader = _load_module("tidformer_dataloader_tmp", osp.join(TIDFORMER_DIR, "utils", "DataLoader.py"))
    return types.SimpleNamespace(
        NeighborSampler=tid_utils.NeighborSampler,
        NegativeEdgeSampler=tid_utils.NegativeEdgeSampler,
        get_idx_data_loader=tid_dataloader.get_idx_data_loader,
        DataClass=tid_dataloader.Data,
        CalendarTimeEncoder=tid_modules.CalendarTimeEncoder,
        DecomposeEncoder=tid_modules.DecomposeEncoder,
        BIEEncoder=tid_model.BIEEncoder,
        TransformerEncoder=tid_model.TransformerEncoder,
    )


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


def time_value(t_norm, t_orig, source):
    return int(t_norm if source == "norm" else t_orig)


def flatten_snapshots(snapshot_list, time_source="raw", label="split"):
    chunks = []
    total = int(len(snapshot_list))
    progress_step = max(1, int(np.ceil(total / 100.0))) if total else 1
    next_progress = progress_step
    t0 = time.perf_counter()
    print(f"[TIDFormer-Fair] flatten {label}: start snapshots={total}", flush=True)
    for idx, (events, t_norm, t_orig) in enumerate(snapshot_list, start=1):
        if len(events) == 0:
            pass
        else:
            t_model = time_value(t_norm, t_orig, time_source)
            t_model_col = np.full((len(events), 1), int(t_model), dtype=np.int64)
            raw_col = np.full((len(events), 1), int(t_orig), dtype=np.int64)
            chunks.append(np.hstack((events.astype(np.int64, copy=False), t_model_col, raw_col)))
        while idx >= next_progress or idx == total:
            print(
                f"[TIDFormer-Fair] flatten {label}: {idx}/{total} "
                f"({100.0 * idx / max(1, total):.1f}%) elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
            if idx >= total:
                break
            next_progress += progress_step
    if not chunks:
        return np.empty((0, 5), dtype=np.int64)
    print(f"[TIDFormer-Fair] flatten {label}: stacking chunks={len(chunks)}", flush=True)
    arr = np.vstack(chunks).astype(np.int64, copy=False)
    if len(arr) > 1 and np.any(arr[1:, 3] < arr[:-1, 3]):
        print(f"[TIDFormer-Fair] flatten {label}: sorting by model time", flush=True)
        arr = arr[np.argsort(arr[:, 3], kind="stable")]
    print(f"[TIDFormer-Fair] flatten {label}: done events={len(arr)} elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    return arr


def build_tidformer_data(tid, train_events, val_events, test_events, num_nodes):
    all_events = np.vstack([x for x in (train_events, val_events, test_events) if len(x)])
    num_edges = len(all_events)
    edge_ids = np.arange(1, num_edges + 1, dtype=np.int64)
    edge_rel_ids = np.zeros(num_edges + 1, dtype=np.int64)
    edge_rel_ids[1:] = all_events[:, 1].astype(np.int64, copy=False) + 1

    src = all_events[:, 0].astype(np.int64, copy=False) + 1
    dst = all_events[:, 2].astype(np.int64, copy=False) + 1
    ts = all_events[:, 3].astype(np.float64, copy=False)
    labels = all_events[:, 1].astype(np.int64, copy=False)

    train_len = len(train_events)
    val_len = len(val_events)
    train_slice = slice(0, train_len)
    val_slice = slice(train_len, train_len + val_len)
    test_slice = slice(train_len + val_len, len(all_events))

    def make_data(slc):
        return tid.DataClass(
            src_node_ids=src[slc],
            dst_node_ids=dst[slc],
            node_interact_times=ts[slc],
            edge_ids=edge_ids[slc],
            labels=labels[slc],
        )

    full_data = tid.DataClass(src, dst, ts, edge_ids, labels)
    train_data = make_data(train_slice)
    val_data = make_data(val_slice)
    test_data = make_data(test_slice)

    node_raw_features = np.zeros((int(num_nodes) + 1, 1), dtype=np.float32)
    return node_raw_features, edge_rel_ids, full_data, train_data, val_data, test_data


def get_neighbor_sampler(tid, data, strategy, time_scaling_factor, seed, num_nodes=None, label=""):
    src_max = int(np.max(data.src_node_ids)) if len(data.src_node_ids) else 0
    dst_max = int(np.max(data.dst_node_ids)) if len(data.dst_node_ids) else 0
    max_node_id = max(src_max, dst_max, int(num_nodes or 0))
    adj_list = [[] for _ in range(max_node_id + 1)]
    label = label or "sampler"
    total_edges = int(len(data.src_node_ids))
    progress_step = max(1, int(np.ceil(total_edges / 100.0))) if total_edges else 1
    next_progress = progress_step
    t0 = time.perf_counter()
    print(
        f"[TIDFormer-Fair] build {label}: start adjacency edges={total_edges} nodes={max_node_id + 1}",
        flush=True,
    )
    for idx, (src, dst, edge_id, ts) in enumerate(
        zip(data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times),
        start=1,
    ):
        adj_list[int(src)].append((int(dst), int(edge_id), float(ts)))
        adj_list[int(dst)].append((int(src), int(edge_id), float(ts)))
        while idx >= next_progress or idx == total_edges:
            print(
                f"[TIDFormer-Fair] build {label}: adjacency {idx}/{total_edges} "
                f"({100.0 * idx / max(1, total_edges):.1f}%) elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
            if idx >= total_edges:
                break
            next_progress += progress_step

    sampler = tid.NeighborSampler.__new__(tid.NeighborSampler)
    sampler.sample_neighbor_strategy = strategy
    sampler.seed = seed
    if strategy == "time_interval_aware":
        sampler.nodes_neighbor_sampled_probabilities = []
        sampler.time_scaling_factor = float(time_scaling_factor)
    sampler.nodes_neighbor_ids = []
    sampler.nodes_edge_ids = []
    sampler.nodes_neighbor_times = []
    if seed is not None:
        sampler.random_state = np.random.RandomState(seed)

    total_nodes = len(adj_list)
    progress_step = max(1, int(np.ceil(total_nodes / 100.0))) if total_nodes else 1
    next_progress = progress_step
    t_sort = time.perf_counter()
    print(f"[TIDFormer-Fair] build {label}: start sorting nodes={total_nodes}", flush=True)
    for node_idx, per_node_neighbors in enumerate(adj_list, start=1):
        sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
        neighbor_times = np.array([x[2] for x in sorted_per_node_neighbors])
        sampler.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
        sampler.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
        sampler.nodes_neighbor_times.append(neighbor_times)
        if strategy == "time_interval_aware":
            sampler.nodes_neighbor_sampled_probabilities.append(
                tid.NeighborSampler.compute_sampled_probabilities(sampler, neighbor_times)
            )
        while node_idx >= next_progress or node_idx == total_nodes:
            print(
                f"[TIDFormer-Fair] build {label}: sorting {node_idx}/{total_nodes} "
                f"({100.0 * node_idx / max(1, total_nodes):.1f}%) elapsed={time.perf_counter() - t_sort:.1f}s",
                flush=True,
            )
            if node_idx >= total_nodes:
                break
            next_progress += progress_step
    print(
        f"[TIDFormer-Fair] build {label}: done total_elapsed={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return sampler


class RelationAwareTIDFormer(nn.Module):
    def __init__(
        self,
        tid,
        node_raw_features,
        edge_relation_ids,
        neighbor_sampler,
        num_rels,
        relation_embedding_dim,
        model_dim,
        time_feat_dim,
        channel_embedding_dim,
        num_layers,
        dropout,
        num_neighbors,
        device,
        num_bidirectional,
        time_segment,
        calendar_base,
        kernel_size,
        bie_feature_dim,
        use_temporal_masking=True,
    ):
        super().__init__()
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_relation_ids = torch.from_numpy(edge_relation_ids.astype(np.int64)).to(device)
        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = int(self.node_raw_features.shape[1])
        self.rel_feat_dim = int(relation_embedding_dim)
        self.model_dim = int(model_dim)
        self.time_feat_dim = int(time_feat_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.num_neighbors = int(num_neighbors)
        self.device = device
        self.bie_feature_dim = int(bie_feature_dim)
        self.num_bidirectional = int(num_bidirectional)
        self.use_neighbor_cache = False
        self._neighbor_cache = {}
        self._neighbor_cache_hits = 0
        self._neighbor_cache_misses = 0

        self.relation_embedding = nn.Embedding(int(num_rels) + 1, self.rel_feat_dim, padding_idx=0)
        self.time_encoder = tid.CalendarTimeEncoder(
            time_dim=time_feat_dim,
            time_segment=time_segment,
            calendar_base=calendar_base,
            parameter_requires_grad=False,
        )
        self.decompose_encoder = tid.DecomposeEncoder(id_dim=time_feat_dim, kernel_size=kernel_size)
        self.bie_encoder = tid.BIEEncoder(
            bie_feat_dim=channel_embedding_dim,
            device=self.device,
            use_temporal_masking=use_temporal_masking,
        )
        self.projection_layer = nn.ModuleDict(
            {
                "node": nn.Linear(self.node_feat_dim, self.model_dim, bias=True),
                "edge": nn.Linear(self.rel_feat_dim, self.model_dim, bias=True),
                "mte": nn.Linear(2 * self.time_feat_dim, self.model_dim, bias=True),
                "ste": nn.Linear(self.time_feat_dim, self.model_dim, bias=True),
                "bie": nn.Linear(channel_embedding_dim, self.model_dim, bias=True),
                "pair": nn.Linear(2 * self.model_dim, self.bie_feature_dim, bias=True),
            }
        )
        self.reduce_layer = nn.Linear(5 * self.model_dim, self.model_dim)
        self.transformers = nn.ModuleList(
            [tid.TransformerEncoder(attention_dim=self.model_dim, num_heads=2, dropout=self.dropout) for _ in range(self.num_layers)]
        )
        self.weightagg = nn.Linear(self.model_dim, 1)
        self.weightagg_pair = nn.Linear(self.bie_feature_dim, 1)

    def set_neighbor_sampler(self, neighbor_sampler):
        self.neighbor_sampler = neighbor_sampler
        self.clear_neighbor_cache()
        if self.neighbor_sampler.sample_neighbor_strategy in ["uniform", "time_interval_aware"]:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def clear_neighbor_cache(self):
        self._neighbor_cache = {}
        self._neighbor_cache_hits = 0
        self._neighbor_cache_misses = 0

    def set_neighbor_cache(self, enabled):
        self.use_neighbor_cache = bool(enabled)
        self.clear_neighbor_cache()

    def neighbor_cache_stats(self):
        return {
            "neighbor_cache_size": int(len(self._neighbor_cache)),
            "neighbor_cache_hits": int(self._neighbor_cache_hits),
            "neighbor_cache_misses": int(self._neighbor_cache_misses),
        }

    def query_relation_features(self, rel_ids):
        rel_ids = torch.as_tensor(rel_ids, dtype=torch.long, device=self.device)
        return self.relation_embedding(rel_ids)

    def get_features(self, node_interact_times, nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times):
        node_idx = torch.from_numpy(nodes_neighbor_ids).long().to(self.device)
        edge_idx = torch.from_numpy(nodes_edge_ids).long().to(self.device)
        mask = torch.from_numpy(nodes_neighbor_ids == 0).to(self.device)

        node_features = self.node_raw_features[node_idx]
        edge_rel_ids = self.edge_relation_ids[edge_idx]
        edge_features = self.relation_embedding(edge_rel_ids)
        time_deltas = torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device)
        time_features = self.time_encoder(timestamps=time_deltas)
        seasonal_features, trend_features = self.decompose_encoder(ids=node_idx.float())

        node_features[mask] = 0.0
        edge_features[mask] = 0.0
        time_features[mask] = 0.0
        seasonal_features[mask] = 0.0
        trend_features[mask] = 0.0
        return node_features, edge_features, time_features, seasonal_features, trend_features

    def get_first_order_historical_neighbors(self, node_ids, node_interact_times):
        if not self.use_neighbor_cache:
            return self.neighbor_sampler.get_first_order_historical_neighbors(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                num_neighbors=self.num_neighbors,
            )

        node_ids = np.asarray(node_ids)
        node_interact_times = np.asarray(node_interact_times)
        neighbor_ids = np.zeros((len(node_ids), self.num_neighbors), dtype=np.longlong)
        edge_ids = np.zeros((len(node_ids), self.num_neighbors), dtype=np.longlong)
        neighbor_times = np.zeros((len(node_ids), self.num_neighbors), dtype=np.float32)
        missing_keys = []
        missing_positions = []
        missing_key_to_index = {}

        for i, (node_id, interact_time) in enumerate(zip(node_ids, node_interact_times)):
            key = (int(node_id), float(interact_time))
            cached = self._neighbor_cache.get(key)
            if cached is not None:
                self._neighbor_cache_hits += 1
                neighbor_ids[i], edge_ids[i], neighbor_times[i] = cached
            else:
                self._neighbor_cache_misses += 1
                missing_idx = missing_key_to_index.get(key)
                if missing_idx is None:
                    missing_idx = len(missing_keys)
                    missing_key_to_index[key] = missing_idx
                    missing_keys.append(key)
                    missing_positions.append([])
                missing_positions[missing_idx].append(i)

        if missing_keys:
            miss_nodes = np.asarray([key[0] for key in missing_keys], dtype=node_ids.dtype)
            miss_times = np.asarray([key[1] for key in missing_keys], dtype=node_interact_times.dtype)
            miss_neighbor_ids, miss_edge_ids, miss_neighbor_times = self.neighbor_sampler.get_first_order_historical_neighbors(
                node_ids=miss_nodes,
                node_interact_times=miss_times,
                num_neighbors=self.num_neighbors,
            )
            for key, positions, n_ids, e_ids, n_times in zip(
                missing_keys, missing_positions, miss_neighbor_ids, miss_edge_ids, miss_neighbor_times
            ):
                cached = (n_ids.copy(), e_ids.copy(), n_times.copy())
                self._neighbor_cache[key] = cached
                for pos in positions:
                    neighbor_ids[pos], edge_ids[pos], neighbor_times[pos] = cached

        return neighbor_ids, edge_ids, neighbor_times

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times):
        src_neighbor_ids, src_edge_ids, src_neighbor_times = self.get_first_order_historical_neighbors(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
        )
        dst_neighbor_ids, dst_edge_ids, dst_neighbor_times = self.get_first_order_historical_neighbors(
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
        )
        src_bie, dst_bie = self.bie_encoder(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            src_nodes_neighbor_ids=src_neighbor_ids,
            dst_nodes_neighbor_ids=dst_neighbor_ids,
            node_interact_times=node_interact_times,
            num_bidirectional=self.num_bidirectional,
        )

        src_node, src_edge, src_time, src_season, src_trend = self.get_features(
            node_interact_times, src_neighbor_ids, src_edge_ids, src_neighbor_times
        )
        dst_node, dst_edge, dst_time, dst_season, dst_trend = self.get_features(
            node_interact_times, dst_neighbor_ids, dst_edge_ids, dst_neighbor_times
        )

        src_node = self.projection_layer["node"](src_node)
        src_edge = self.projection_layer["edge"](src_edge)
        src_time = self.projection_layer["mte"](src_time)
        src_decompose = self.projection_layer["ste"](torch.cat([src_season, src_trend], dim=-1))
        src_bie = self.projection_layer["bie"](src_bie)

        dst_node = self.projection_layer["node"](dst_node)
        dst_edge = self.projection_layer["edge"](dst_edge)
        dst_time = self.projection_layer["mte"](dst_time)
        dst_decompose = self.projection_layer["ste"](torch.cat([dst_season, dst_trend], dim=-1))
        dst_bie = self.projection_layer["bie"](dst_bie)

        src_combined = self.reduce_layer(torch.cat([src_node, src_edge, src_time, src_decompose, src_bie], dim=-1))
        dst_combined = self.reduce_layer(torch.cat([dst_node, dst_edge, dst_time, dst_decompose, dst_bie], dim=-1))

        for transformer in self.transformers:
            src_combined = transformer(src_combined)
        for transformer in self.transformers:
            dst_combined = transformer(dst_combined)

        src_weight = self.weightagg(src_combined).transpose(1, 2)
        dst_weight = self.weightagg(dst_combined).transpose(1, 2)
        src_embedding = src_weight.matmul(src_combined).squeeze(dim=1)
        dst_embedding = dst_weight.matmul(dst_combined).squeeze(dim=1)

        bie_pair = self.projection_layer["pair"](torch.cat([src_bie, dst_bie], dim=2))
        pair_weight = self.weightagg_pair(bie_pair).transpose(1, 2)
        bie_pair = pair_weight.matmul(bie_pair).squeeze(dim=1)
        return src_embedding, dst_embedding, bie_pair


class RelationAwareMergeLayer(nn.Module):
    def __init__(self, node_dim, relation_dim, bie_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(2 * int(node_dim) + int(relation_dim) + int(bie_dim), int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), int(output_dim))
        self.act = nn.ReLU()

    def forward(self, src_embedding, dst_embedding, bie_features, query_relation_features):
        x = torch.cat([src_embedding, dst_embedding, bie_features, query_relation_features], dim=1)
        return self.fc2(self.act(self.fc1(x)))


class RelationAwareTIDFormerModel(nn.Module):
    def __init__(self, backbone, predictor):
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor

    def set_neighbor_sampler(self, neighbor_sampler):
        self.backbone.set_neighbor_sampler(neighbor_sampler)

    def score_logits(self, src_node_ids, dst_node_ids, node_interact_times, relation_ids):
        src_emb, dst_emb, bie = self.backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
        )
        rel_emb = self.backbone.query_relation_features(relation_ids)
        return self.predictor(src_emb, dst_emb, bie, rel_emb).squeeze(dim=-1)

    def score_probabilities(self, src_node_ids, dst_node_ids, node_interact_times, relation_ids):
        return torch.sigmoid(self.score_logits(src_node_ids, dst_node_ids, node_interact_times, relation_ids))


def make_out_dir(args):
    name = (
        f"nsq{args.ns_q}_ns{args.ns_seed}_tpr{args.train_predict_ratio:g}"
        f"_md{args.model_dim}_rd{args.relation_dim}_td{args.time_feat_dim}"
        f"_cd{args.channel_embedding_dim}_ly{args.num_layers}_neigh{args.num_neighbors}"
        f"_bs{args.batch_size}_lr{args.learning_rate:g}"
    )
    return osp.join("results_tidformer_fair", args.dataset, f"seed{args.seed}", name)


def serializable_args(args):
    config = {}
    for key, value in vars(args).items():
        if key.startswith("_"):
            continue
        if key == "device":
            config[key] = str(value)
        elif isinstance(value, (np.integer,)):
            config[key] = int(value)
        elif isinstance(value, (np.floating,)):
            config[key] = float(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value
    return config


def build_model(args, tid, node_raw_features, edge_rel_ids, train_neighbor_sampler, num_rels):
    backbone = RelationAwareTIDFormer(
        tid=tid,
        node_raw_features=node_raw_features,
        edge_relation_ids=edge_rel_ids,
        neighbor_sampler=train_neighbor_sampler,
        num_rels=int(num_rels),
        relation_embedding_dim=int(args.relation_dim),
        model_dim=int(args.model_dim),
        time_feat_dim=int(args.time_feat_dim),
        channel_embedding_dim=int(args.channel_embedding_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        num_neighbors=int(args.num_neighbors),
        device=args.device,
        num_bidirectional=int(args.num_bidirectional),
        time_segment=int(args.num_time_segment),
        calendar_base=args.calendar_base,
        kernel_size=int(args.kernel_size),
        bie_feature_dim=int(args.bie_feature_dim),
        use_temporal_masking=bool(args.use_temporal_masking),
    )
    predictor = RelationAwareMergeLayer(
        node_dim=int(args.model_dim),
        relation_dim=int(args.relation_dim),
        bie_dim=int(args.bie_feature_dim),
        hidden_dim=int(args.predictor_hidden_dim or args.model_dim),
    )
    return RelationAwareTIDFormerModel(backbone, predictor).to(args.device)


def train_one_epoch(args, model, train_data, train_loader, train_neg_sampler, optimizer, loss_func, epoch=None):
    model.train()
    model.set_neighbor_sampler(args._train_neighbor_sampler)
    losses = []
    train_time = 0.0
    num_batches = int(len(train_loader))
    progress_step = max(1, int(np.ceil(num_batches / 100.0))) if num_batches else 1
    next_progress = progress_step
    wall_t0 = time.perf_counter()
    epoch_label = f"epoch={epoch}" if epoch is not None else "epoch=?"

    for batch_idx, train_indices in enumerate(train_loader, start=1):
        sync_device(args.device)
        t0 = time.perf_counter()
        idx = train_indices.numpy()
        batch_src = train_data.src_node_ids[idx]
        batch_dst = train_data.dst_node_ids[idx]
        batch_times = train_data.node_interact_times[idx]
        batch_rel = train_data.labels[idx].astype(np.int64, copy=False) + 1
        _, batch_neg_dst = train_neg_sampler.sample(
            size=len(batch_src),
            batch_src_node_ids=batch_src,
            batch_dst_node_ids=batch_dst,
            current_batch_start_time=float(np.min(batch_times)),
            current_batch_end_time=float(np.max(batch_times)),
        )
        batch_neg_src = batch_src
        batch_neg_rel = batch_rel

        pos_prob = model.score_probabilities(batch_src, batch_dst, batch_times, batch_rel)
        neg_prob = model.score_probabilities(batch_neg_src, batch_neg_dst, batch_times, batch_neg_rel)
        predicts = torch.cat([pos_prob, neg_prob], dim=0)
        labels = torch.cat([torch.ones_like(pos_prob), torch.zeros_like(neg_prob)], dim=0)
        loss = loss_func(input=predicts, target=labels)

        optimizer.zero_grad()
        loss.backward()
        if float(args.grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_norm))
        optimizer.step()
        sync_device(args.device)
        train_time += time.perf_counter() - t0
        losses.append(float(loss.detach().cpu().item()))
        while batch_idx >= next_progress or batch_idx == num_batches:
            print(
                f"[TIDFormer-Fair] train {epoch_label}: batch {batch_idx}/{num_batches} "
                f"({100.0 * batch_idx / max(1, num_batches):.1f}%) "
                f"loss={losses[-1]:.5f} elapsed={time.perf_counter() - wall_t0:.1f}s",
                flush=True,
            )
            if batch_idx >= num_batches:
                break
            next_progress += progress_step

    return {"loss": float(np.mean(losses)) if losses else 0.0, "train_time_s": float(train_time)}


@torch.no_grad()
def score_pairs(args, model, src, dst, times, rel, measure_forward=False):
    scores = []
    forward_time = 0.0
    pair_batch_size = 1 if bool(args.exact_eval) else int(args.eval_pair_batch_size)
    for start in range(0, len(src), pair_batch_size):
        end = start + pair_batch_size
        sync_device(args.device)
        t0 = time.perf_counter()
        prob = model.score_probabilities(src[start:end], dst[start:end], times[start:end], rel[start:end])
        sync_device(args.device)
        if measure_forward:
            forward_time += time.perf_counter() - t0
        scores.append(prob.detach().cpu().numpy().astype(np.float32, copy=False))
    if not scores:
        return np.empty(0, dtype=np.float32), forward_time
    return np.concatenate(scores), forward_time


@torch.no_grad()
def evaluate_split(args, model, split_name, snapshot_list, measure_forward=False, max_negatives=None, profile_name=None):
    model.eval()
    model.set_neighbor_sampler(args._full_neighbor_sampler)
    neg_sampler = args._negative_sampler
    sums = {}
    forward_time = 0.0
    sample_count = 0
    neighbor_cache_hits = 0
    neighbor_cache_misses = 0
    max_neighbor_cache_size = 0
    total_queries = int(sum(len(events) for events, _, _ in snapshot_list))
    progress_step = max(1, int(np.ceil(total_queries / 100.0))) if total_queries else 1
    next_progress = progress_step
    eval_t0 = time.perf_counter()
    label = profile_name or split_name
    effective_max_negs = None if max_negatives is None or int(max_negatives) <= 0 else int(max_negatives)

    print(
        f"[TIDFormer-Fair] eval {label}: start queries={total_queries} "
        f"ns_q={args.ns_q} max_negs={effective_max_negs or 'full'} "
        f"eval_batch={args.eval_batch_size} pair_batch={1 if args.exact_eval else args.eval_pair_batch_size} "
        f"neg_cols={1 if args.exact_eval else args.eval_neg_columns_per_call} exact_eval={args.exact_eval}",
        flush=True,
    )

    model.backbone.set_neighbor_cache(bool(args.eval_neighbor_cache))
    for events, t_norm, raw_t in snapshot_list:
        if len(events) == 0:
            continue
        model.backbone.clear_neighbor_cache()
        t_model = time_value(t_norm, raw_t, args.time_source)
        for batch, neg_arr, neg_mask in collect_eval_batch(events[:, :3], int(raw_t), neg_sampler, split_name, args.eval_batch_size):
            if len(batch) == 0:
                continue
            if effective_max_negs is not None and neg_arr.shape[1] > effective_max_negs:
                neg_arr = neg_arr[:, :effective_max_negs]
                neg_mask = neg_mask[:, :effective_max_negs]
            bsz, neg_width = neg_arr.shape
            src = batch[:, 0].astype(np.int64, copy=False) + 1
            dst = batch[:, 2].astype(np.int64, copy=False) + 1
            rel = batch[:, 1].astype(np.int64, copy=False) + 1
            times = np.full(bsz, float(t_model), dtype=np.float64)

            pos_scores, dt = score_pairs(args, model, src, dst, times, rel, measure_forward=measure_forward)
            forward_time += dt
            pos_scores = pos_scores.reshape(-1, 1)
            neg_scores = np.zeros((bsz, neg_width), dtype=np.float32)

            neg_columns_per_call = 1 if bool(args.exact_eval) else int(args.eval_neg_columns_per_call)
            for col_start in range(0, neg_width, neg_columns_per_call):
                col_end = min(neg_width, col_start + neg_columns_per_call)
                sub_mask = neg_mask[:, col_start:col_end]
                if not np.any(sub_mask):
                    continue
                row_idx, col_idx = np.nonzero(sub_mask)
                flat_dst = neg_arr[row_idx, col_start + col_idx].astype(np.int64, copy=False) + 1
                flat_src = src[row_idx]
                flat_rel = rel[row_idx]
                flat_times = times[row_idx]
                flat_scores, dt = score_pairs(
                    args,
                    model,
                    flat_src,
                    flat_dst,
                    flat_times,
                    flat_rel,
                    measure_forward=measure_forward,
                )
                forward_time += dt
                neg_scores[row_idx, col_start + col_idx] = flat_scores

            add_metric_sums(sums, compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask))
            sample_count += int(bsz)
            while sample_count >= next_progress or sample_count == total_queries:
                elapsed = time.perf_counter() - eval_t0
                pct = 100.0 * sample_count / max(1, total_queries)
                print(
                    f"[TIDFormer-Fair] eval {label}: completed {sample_count}/{total_queries} "
                    f"({pct:.1f}%) elapsed={elapsed:.1f}s",
                    flush=True,
                )
                if sample_count >= total_queries:
                    break
                next_progress += progress_step
        cache_stats = model.backbone.neighbor_cache_stats()
        neighbor_cache_hits += int(cache_stats["neighbor_cache_hits"])
        neighbor_cache_misses += int(cache_stats["neighbor_cache_misses"])
        max_neighbor_cache_size = max(max_neighbor_cache_size, int(cache_stats["neighbor_cache_size"]))
    model.backbone.set_neighbor_cache(False)

    metrics = finalize_metric_sums(sums)
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    profile = {
        "forward_time_s": float(forward_time),
        "sample_count": int(sample_count),
        "max_negatives": int(effective_max_negs) if effective_max_negs is not None else None,
        "wall_time_s": float(time.perf_counter() - eval_t0),
        "exact_eval": bool(args.exact_eval),
        "eval_neighbor_cache": bool(args.eval_neighbor_cache),
        "neighbor_cache_hits": int(neighbor_cache_hits),
        "neighbor_cache_misses": int(neighbor_cache_misses),
        "max_neighbor_cache_size": int(max_neighbor_cache_size),
    }
    print(
        f"[TIDFormer-Fair] eval {label}: done mrr={metrics['mrr_strict']:.6f} "
        f"hit1={metrics['hit@1_strict']:.6f} hit10={metrics['hit@10_strict']:.6f} "
        f"wall={profile['wall_time_s']:.1f}s forward={forward_time:.1f}s "
        f"neighbor_cache_hits={neighbor_cache_hits} misses={neighbor_cache_misses} "
        f"max_cache={max_neighbor_cache_size}",
        flush=True,
    )
    return metrics, profile


def run(args):
    tid = import_tidformer_official()
    set_random_seed(args.seed)
    if getattr(args, "disable_cudnn", True):
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        print(
            f"[TIDFormer-Fair] disabled cuDNN backend for temporal attention stability "
            f"(cudnn.enabled={torch.backends.cudnn.enabled}).",
            flush=True,
        )

    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and int(args.gpu) >= 0 else "cpu")
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)

    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[TIDFormer-Fair]")
    args._negative_sampler = data["negative_sampler"]

    train_events = flatten_snapshots(data["train_list"], time_source=args.time_source, label="train")
    val_events = flatten_snapshots(data["val_list"], time_source=args.time_source, label="val")
    test_events = flatten_snapshots(data["test_list"], time_source=args.time_source, label="test")
    print("[TIDFormer-Fair] build tidformer data: start", flush=True)
    t_data_build = time.perf_counter()
    node_raw_features, edge_rel_ids, full_data, train_data, _, _ = build_tidformer_data(
        tid, train_events, val_events, test_events, int(data["num_nodes"])
    )
    print(
        f"[TIDFormer-Fair] build tidformer data: done total_events={len(full_data.src_node_ids)} "
        f"elapsed={time.perf_counter() - t_data_build:.1f}s",
        flush=True,
    )

    args._train_neighbor_sampler = get_neighbor_sampler(
        tid,
        train_data,
        args.sample_neighbor_strategy,
        args.time_scaling_factor,
        seed=0,
        num_nodes=int(data["num_nodes"]),
        label="train_neighbor_sampler",
    )
    args._full_neighbor_sampler = get_neighbor_sampler(
        tid,
        full_data,
        args.sample_neighbor_strategy,
        args.time_scaling_factor,
        seed=1,
        num_nodes=int(data["num_nodes"]),
        label="full_neighbor_sampler",
    )
    train_neg_sampler = tid.NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids,
        dst_node_ids=train_data.dst_node_ids,
        interact_times=train_data.node_interact_times,
        seed=None,
    )
    train_loader = tid.get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=int(args.batch_size),
        shuffle=False,
    )

    out_dir = make_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    save_config(out_dir, serializable_args(args))

    model = build_model(args, tid, node_raw_features, edge_rel_ids, args._train_neighbor_sampler, int(data["num_rels"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    loss_func = nn.BCELoss()

    print(
        f"[TIDFormer-Fair] model nodes={data['num_nodes']} rels={data['num_rels']} "
        f"internal_nodes={data['num_nodes'] + 1} rel_dim={args.relation_dim} "
        f"model_dim={args.model_dim} device={args.device} time_source={args.time_source} "
        f"evaluate_every={args.evaluate_every} intermediate_val_ns={args.intermediate_val_ns_q} "
        f"exact_eval={args.exact_eval} eval_neighbor_cache={args.eval_neighbor_cache}",
        flush=True,
    )

    reset_cuda_peak(args.device)
    val_checkpoint_path = osp.join(out_dir, "best_val_model.pt")
    loss_checkpoint_path = osp.join(out_dir, "best_train_loss_model.pt")
    best_val = -float("inf")
    best_val_epoch = 0
    best_train_loss = float("inf")
    best_train_loss_epoch = 0
    early_stopped = False
    early_stop_epoch = 0
    train_time_total = 0.0
    epoch_logs = []

    for epoch in range(1, int(args.num_epochs) + 1):
        log = train_one_epoch(args, model, train_data, train_loader, train_neg_sampler, optimizer, loss_func, epoch=epoch)
        train_time_total += float(log["train_time_s"])
        log["epoch"] = int(epoch)
        if float(log["loss"]) < best_train_loss - float(args.tolerance):
            best_train_loss = float(log["loss"])
            best_train_loss_epoch = int(epoch)
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "train_loss": best_train_loss},
                loss_checkpoint_path,
            )
        log["best_train_loss"] = float(best_train_loss)
        log["epochs_since_best_train_loss"] = int(epoch - best_train_loss_epoch)

        do_val = int(args.evaluate_every) > 0 and epoch % int(args.evaluate_every) == 0
        if do_val:
            val_metrics, val_profile = evaluate_split(
                args,
                model,
                "val",
                data["val_list"],
                measure_forward=False,
                max_negatives=int(args.intermediate_val_ns_q),
                profile_name="val-intermediate",
            )
            log["val_mrr_strict"] = float(val_metrics["mrr_strict"])
            log["val_hit@1_strict"] = float(val_metrics["hit@1_strict"])
            log["val_hit@10_strict"] = float(val_metrics["hit@10_strict"])
            log["val_max_negatives"] = val_profile["max_negatives"]
            if val_metrics["mrr_strict"] > best_val + float(args.tolerance):
                best_val = float(val_metrics["mrr_strict"])
                best_val_epoch = int(epoch)
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, val_checkpoint_path)
            log["epochs_since_best_val"] = int(epoch - best_val_epoch) if best_val_epoch else 0
        epoch_logs.append(log)
        print(
            f"[TIDFormer-Fair] epoch={epoch} loss={log['loss']:.5f} "
            f"train_time={log['train_time_s']:.2f}s "
            f"best_train_loss={best_train_loss:.5f}@{best_train_loss_epoch} "
            f"best_val_mrr={max(best_val, 0.0):.5f}",
            flush=True,
        )
        if best_train_loss_epoch and int(epoch - best_train_loss_epoch) >= int(args.patience):
            early_stopped = True
            early_stop_epoch = int(epoch)
            print(
                f"[TIDFormer-Fair] early stop at epoch={epoch}: train loss did not improve for "
                f"{epoch - best_train_loss_epoch} epochs "
                f"(patience={args.patience}, best_train_loss_epoch={best_train_loss_epoch})",
                flush=True,
            )
            break

    train_peak = cuda_peak_allocated(args.device)
    selected_by = "val_mrr" if best_val_epoch and osp.exists(val_checkpoint_path) else "train_loss"
    selected_path = val_checkpoint_path if selected_by == "val_mrr" else loss_checkpoint_path
    if osp.exists(selected_path):
        ckpt = torch.load(selected_path, map_location=args.device)
        model.load_state_dict(ckpt["state_dict"])
        best_epoch = int(ckpt.get("epoch", best_val_epoch or best_train_loss_epoch))
    else:
        best_epoch = int(epoch_logs[-1]["epoch"]) if epoch_logs else 0
        selected_by = "final"
        torch.save({"state_dict": model.state_dict(), "epoch": best_epoch, "val_metrics": {}}, loss_checkpoint_path)

    val_metrics, val_profile = evaluate_split(args, model, "val", data["val_list"], measure_forward=False)
    reset_cuda_peak(args.device)
    test_metrics, test_profile = evaluate_split(args, model, "test", data["test_list"], measure_forward=True)
    eval_peak = cuda_peak_allocated(args.device)
    reported_best_val = float(best_val) if np.isfinite(best_val) else float(val_metrics["mrr_strict"])

    metrics = {
        "format": "tidformer_fair_v1",
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
        "early_stop_metric": "train_loss",
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
            "Official TIDFormer temporal neighbor sampling, mixed-granularity time encoding, "
            "season/trend decomposition, BIE, Transformer aggregation, BCE training with one "
            "negative per positive, and recent/uniform/time-aware neighbor strategies are retained. "
            "This fair adapter offsets node ids because official TIDFormer reserves 0 for padding, "
            "uses learnable relation embeddings for historical edge features, and conditions the "
            "final predictor on the current query relation. Final val/test ranking scores exactly "
            "one positive plus the protocol negatives. The fast default batches candidate pairs "
            "using TIDFormer's official batch-style BIE; pass --exact-eval to score pairs one at "
            "a time and avoid candidate coupling at much higher cost."
        ),
    }
    save_metrics(out_dir, metrics)
    print(
        f"[TIDFormer-Fair] final val_mrr={metrics['val_mrr']:.6f} "
        f"test_mrr={metrics['test_mrr']:.6f} test_hit1={metrics['test_hit1']:.6f} "
        f"test_hit10={metrics['test_hit10']:.6f} selected={selected_by}@epoch{best_epoch}",
        flush=True,
    )
    print(
        f"[TIDFormer-Fair] train_time={train_time_total:.3f}s "
        f"test_forward_time={test_profile['forward_time_s']:.3f}s "
        f"test_samples={test_profile['sample_count']} "
        f"train_peak={format_bytes(train_peak)} eval_peak={format_bytes(eval_peak)} saved -> {out_dir}",
        flush=True,
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser("Fair relation-aware TIDFormer trainer for EAGLE TKG/THG protocols.")
    parser.add_argument("--dataset", type=str, required=True, choices=eagle_utils.SUPPORTED_DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--disable-cudnn", action="store_true", default=True)
    parser.add_argument("--enable-cudnn", dest="disable_cudnn", action="store_false")

    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_pair_batch_size", type=int, default=256)
    parser.add_argument("--eval_neg_columns_per_call", type=int, default=256)
    parser.add_argument("--exact-eval", action="store_true", default=False)
    parser.add_argument("--eval-neighbor-cache", action="store_true", default=True)
    parser.add_argument("--no-eval-neighbor-cache", dest="eval_neighbor_cache", action="store_false")
    parser.add_argument("--intermediate-val-ns-q", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_norm", type=float, default=0.0)

    parser.add_argument("--num_neighbors", type=int, default=32)
    parser.add_argument("--sample_neighbor_strategy", type=str, default="recent", choices=("uniform", "recent", "time_interval_aware"))
    parser.add_argument("--time_scaling_factor", type=float, default=1e-6)
    parser.add_argument("--time_source", type=str, default="raw", choices=("raw", "norm"))
    parser.add_argument("--time_feat_dim", type=int, default=100)
    parser.add_argument("--num_time_segment", type=int, default=4)
    parser.add_argument("--calendar_base", type=str, default="yearly", choices=("weekly", "monthly", "yearly", "none"))
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--model_dim", type=int, default=172)
    parser.add_argument("--relation_dim", type=int, default=64)
    parser.add_argument("--channel_embedding_dim", type=int, default=50)
    parser.add_argument("--bie_feature_dim", type=int, default=8)
    parser.add_argument("--predictor_hidden_dim", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_bidirectional", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-temporal-masking", action="store_true", default=True)
    parser.add_argument("--no-temporal-masking", dest="use_temporal_masking", action="store_false")

    args = parser.parse_args()
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    for name in ("batch_size", "eval_batch_size", "eval_pair_batch_size", "eval_neg_columns_per_call"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if int(args.num_epochs) <= 0 or int(args.patience) <= 0:
        raise ValueError("--num_epochs and --patience must be positive")
    if int(args.evaluate_every) == 0 or int(args.evaluate_every) < -1:
        raise ValueError("--evaluate-every must be -1 or a positive integer")
    if int(args.intermediate_val_ns_q) < 0:
        raise ValueError("--intermediate-val-ns-q must be non-negative")
    if int(args.num_neighbors) <= 0 or int(args.num_layers) <= 0:
        raise ValueError("--num_neighbors and --num_layers must be positive")
    if int(args.model_dim) <= 0 or int(args.relation_dim) <= 0:
        raise ValueError("--model_dim and --relation_dim must be positive")
    if int(args.model_dim) % 2 != 0:
        raise ValueError("--model_dim must be divisible by 2 because TIDFormer's Transformer uses 2 heads")
    if int(args.time_feat_dim) <= 0 or int(args.channel_embedding_dim) <= 0:
        raise ValueError("--time_feat_dim and --channel_embedding_dim must be positive")
    if int(args.time_feat_dim) % 2 != 0:
        raise ValueError("--time_feat_dim must be even because TIDFormer's DecomposeEncoder splits it in half")
    if int(args.kernel_size) <= 0 or int(args.kernel_size) % 2 == 0:
        raise ValueError("--kernel_size must be a positive odd integer")
    return args


if __name__ == "__main__":
    run(parse_args())
