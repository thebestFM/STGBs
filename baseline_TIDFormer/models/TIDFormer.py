import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import CalendarTimeEncoder, DecomposeEncoder
from utils.utils import NeighborSampler


class TIDFormer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, num_layers: int = 2,
                 dropout: float = 0.1, num_neighbors: int = 32, device: str = 'cpu',
                 num_bidirectional: int = 2, time_segment: int = 4, calendar_base: str = 'weekly', kernel_size: int = 5,
                 BIE_feature_dim: int = 16, use_temporal_masking: bool = True):
        """
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param dropout: float, dropout rate
        :param num_neighbors: int, number of neighbors to sample for each node
        :param device: str, device
        :param num_bidirectional: int, number of bidirectional interactions
        :param num_time_segment: int, number of time segments
        :param calendar_base: str, calendar-based information to use
        :param kernel_size: int, kernel size of moving average
        :param use_temporal_masking: bool, whether to prevent batch internal information leakage
        """
        super(TIDFormer, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_neighbors = num_neighbors
        self.device = device
        self.BIE_feature_dim = BIE_feature_dim
        self.use_temporal_masking = use_temporal_masking

        # Mixed-granularity Temporal Encoding
        self.time_encoder = CalendarTimeEncoder(time_dim=time_feat_dim, time_segment=time_segment, calendar_base=calendar_base, parameter_requires_grad=False)
        
        # Seasonality & Trend Encoding
        self.decompose_encoder = DecomposeEncoder(id_dim=time_feat_dim, kernel_size=kernel_size)

        # Bidirectional Interaction Encoding
        self.bie_feat_dim = self.channel_embedding_dim
        self.bie_encoder = BIEEncoder(bie_feat_dim=self.bie_feat_dim, device=self.device, use_temporal_masking=self.use_temporal_masking)
        self.num_bidirectional = num_bidirectional

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'mte': nn.Linear(in_features=2*self.time_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'ste': nn.Linear(in_features=self.time_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'bie': nn.Linear(in_features=self.bie_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'pair': nn.Linear(in_features=2*self.edge_feat_dim, out_features=self.BIE_feature_dim, bias=True)
        })
        self.reduce_layer = nn.Linear(5*self.edge_feat_dim, self.edge_feat_dim)
       
        # Transformer
        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.edge_feat_dim, num_heads=2, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])
       
        self.weightagg = nn.Linear(self.edge_feat_dim, 1)
        self.weightagg_pair = nn.Linear(self.BIE_feature_dim, 1)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, 
                                                node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        src_nodes_neighbor_ids, src_nodes_edge_ids, src_nodes_neighbor_times = \
           self.neighbor_sampler.get_first_order_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.num_neighbors)

        dst_nodes_neighbor_ids, dst_nodes_edge_ids, dst_nodes_neighbor_times = \
            self.neighbor_sampler.get_first_order_historical_neighbors(node_ids=dst_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.num_neighbors)
        
        # get bidirectional interaction embeddings
        src_nodes_bie_features, dst_nodes_bie_features = \
            self.bie_encoder(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                           src_nodes_neighbor_ids=src_nodes_neighbor_ids,
                           dst_nodes_neighbor_ids=dst_nodes_neighbor_ids, 
                           node_interact_times=node_interact_times,
                           num_bidirectional=self.num_bidirectional)

        # get node & edge embeddings, mixed-granularity temporal embeddings, and seasonality & trend embeddings
        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features, src_nodes_neighbor_seasonal_features, src_nodes_neighbor_trend_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=src_nodes_neighbor_ids,
                              nodes_edge_ids=src_nodes_edge_ids, nodes_neighbor_times=src_nodes_neighbor_times, time_encoder=self.time_encoder, decompose_encoder=self.decompose_encoder)
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, dst_nodes_neighbor_seasonal_features, dst_nodes_neighbor_trend_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=dst_nodes_neighbor_ids,
                              nodes_edge_ids=dst_nodes_edge_ids, nodes_neighbor_times=dst_nodes_neighbor_times, time_encoder=self.time_encoder, decompose_encoder=self.decompose_encoder)
        
        src_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_nodes_neighbor_node_raw_features)
        src_nodes_edge_raw_features = self.projection_layer['edge'](src_nodes_edge_raw_features)
        src_nodes_neighbor_time_features = self.projection_layer['mte'](src_nodes_neighbor_time_features)
        src_nodes_decompose_features = self.projection_layer['ste'](torch.cat([src_nodes_neighbor_seasonal_features, src_nodes_neighbor_trend_features], dim=-1))
        src_nodes_bie_features = self.projection_layer['bie'](src_nodes_bie_features)

        dst_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_nodes_neighbor_node_raw_features)
        dst_nodes_edge_raw_features = self.projection_layer['edge'](dst_nodes_edge_raw_features)
        dst_nodes_neighbor_time_features = self.projection_layer['mte'](dst_nodes_neighbor_time_features)
        dst_nodes_decompose_features = self.projection_layer['ste'](torch.cat([dst_nodes_neighbor_seasonal_features, dst_nodes_neighbor_trend_features], dim=-1))
        dst_nodes_bie_features = self.projection_layer['bie'](dst_nodes_bie_features)
        
        src_combined_features = torch.cat([src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features, src_nodes_decompose_features, src_nodes_bie_features], dim=-1)
        dst_combined_features = torch.cat([dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, dst_nodes_decompose_features, dst_nodes_bie_features], dim=-1)

        src_combined_features = self.reduce_layer(src_combined_features)
        dst_combined_features = self.reduce_layer(dst_combined_features)
        
        for transformer in self.transformers:
            src_combined_features = transformer(src_combined_features)
        for transformer in self.transformers:
            dst_combined_features = transformer(dst_combined_features)
        
        src_weight = self.weightagg(src_combined_features).transpose(1, 2)
        dst_weight = self.weightagg(dst_combined_features).transpose(1, 2)
       
        src_combined_features = src_weight.matmul(src_combined_features).squeeze(dim=1)
        dst_combined_features = dst_weight.matmul(dst_combined_features).squeeze(dim=1)

        BIE_features = self.projection_layer['pair'](torch.cat([src_nodes_bie_features, dst_nodes_bie_features], dim=2))
        pair_weight = self.weightagg_pair(BIE_features).transpose(1, 2)
        BIE_features = pair_weight.matmul(BIE_features).squeeze(dim=1)
        
        return src_combined_features, dst_combined_features, BIE_features

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray, time_encoder: CalendarTimeEncoder, decompose_encoder: DecomposeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: CalendarTimeEncoder, time encoder
        :param decompose_encoder: DecomposeEncoder, seasonality & trend encoder
        :return:
        """
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))
        nodes_neighbor_seasonal_features, nodes_neighbor_trend_features = decompose_encoder(ids=torch.from_numpy(nodes_neighbor_ids).float().to(self.device))
        
        nodes_neighbor_time_features[torch.from_numpy(nodes_neighbor_ids == 0)] = 0.0
        nodes_neighbor_seasonal_features[torch.from_numpy(nodes_neighbor_ids == 0)] = 0.0
        nodes_neighbor_trend_features[torch.from_numpy(nodes_neighbor_ids == 0)] = 0.0

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features, nodes_neighbor_seasonal_features, nodes_neighbor_trend_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class BIEEncoder(nn.Module):

    def __init__(self, bie_feat_dim: int, device: str = 'cpu', use_temporal_masking: bool = True):
        super(BIEEncoder, self).__init__()

        self.bie_feat_dim = bie_feat_dim
        self.device = device
        self.use_temporal_masking = use_temporal_masking

        self.src_bie_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.bie_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.bie_feat_dim, out_features=self.bie_feat_dim))

        self.dst_bie_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.bie_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.bie_feat_dim, out_features=self.bie_feat_dim))

    def count_nodes_appearances(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, 
                              src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray, 
                              node_interact_times: np.ndarray, num_bidirectional: int):
        """
        Count node appearances with minimal temporal masking to prevent information leakage
        """
        # two lists to store the appearances of source and destination nodes
        src_nodes_appearances, dst_nodes_appearances = [], []

        # Create temporal mask if needed
        if self.use_temporal_masking and node_interact_times is not None:
            # Sort indices by time to apply temporal masking
            time_sorted_indices = np.argsort(node_interact_times)
            # Create a mask where each position can only see previous positions
            temporal_mask = np.tril(np.ones((len(src_node_ids), len(src_node_ids)), dtype=bool), k=-1)
        else:
            # No temporal masking - use original behavior
            temporal_mask = np.ones((len(src_node_ids), len(src_node_ids)), dtype=bool)
            time_sorted_indices = np.arange(len(src_node_ids))

        for i in range(len(src_node_ids)):
            curr_idx = time_sorted_indices[i] if self.use_temporal_masking else i
            src_node_id = src_node_ids[curr_idx]
            dst_node_id = dst_node_ids[curr_idx]
            src_node_neighbor_ids = src_nodes_neighbor_ids[curr_idx]
            dst_node_neighbor_ids = dst_nodes_neighbor_ids[curr_idx]

            # Calculate unique keys and counts for source and destination
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_node_neighbor_ids, return_inverse=True, return_counts=True)
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_node_neighbor_ids, return_inverse=True, return_counts=True)
            
            # Create mappings from node IDs to their counts
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # Adjust counts specifically for the cases where src_node_id appears in dst's neighbors and vice versa
            if src_node_id in dst_mapping_dict:
                src_count_in_dst = dst_mapping_dict[src_node_id]
                src_mapping_dict[src_node_id] = src_count_in_dst
                dst_mapping_dict[src_node_id] = src_count_in_dst
            if dst_node_id in src_mapping_dict:
                dst_count_in_src = src_mapping_dict[dst_node_id]
                src_mapping_dict[dst_node_id] = dst_count_in_src
                dst_mapping_dict[dst_node_id] = dst_count_in_src
            
            # Bidirectional Interaction with temporal masking
            dst_unique_keys_temp = dst_unique_keys.tolist()
            src_unique_keys_temp = src_unique_keys.tolist()
            if 0 in dst_unique_keys_temp:
                dst_unique_keys_temp.remove(0)
            if src_node_id in dst_unique_keys_temp:
                dst_unique_keys_temp.remove(src_node_id)

            if 0 in src_unique_keys_temp:
                src_unique_keys_temp.remove(0)
            if dst_node_id in src_unique_keys_temp:
                src_unique_keys_temp.remove(dst_node_id)

            # Apply temporal masking to bidirectional interactions
            if self.use_temporal_masking:
                # Only consider events that happened before current event
                visible_src_indices = time_sorted_indices[:i]  # Events before current
                visible_dst_indices = time_sorted_indices[:i]
                visible_src_node_ids = src_node_ids[visible_src_indices] if len(visible_src_indices) > 0 else np.array([])
                visible_dst_node_ids = dst_node_ids[visible_dst_indices] if len(visible_dst_indices) > 0 else np.array([])
                visible_src_neighbors = src_nodes_neighbor_ids[visible_src_indices] if len(visible_src_indices) > 0 else np.array([]).reshape(0, src_nodes_neighbor_ids.shape[1])
                visible_dst_neighbors = dst_nodes_neighbor_ids[visible_dst_indices] if len(visible_dst_indices) > 0 else np.array([]).reshape(0, dst_nodes_neighbor_ids.shape[1])
            else:
                # Original behavior - can see all events in batch
                visible_src_node_ids = src_node_ids
                visible_dst_node_ids = dst_node_ids
                visible_src_neighbors = src_nodes_neighbor_ids
                visible_dst_neighbors = dst_nodes_neighbor_ids

            # Reconstruction for dst (using only visible events)
            dst_id_neighbor_ids = set()
            if len(dst_unique_keys_temp) != 0 and len(visible_src_node_ids) > 0:
                for node_id in dst_unique_keys_temp:
                    if node_id in visible_src_node_ids:
                        index = np.where(visible_src_node_ids == node_id)
                        if len(index[0]) > 0:
                            neighbor_idx = index[0][0]  # Take first occurrence
                            dst_id_neighbor_ids = dst_id_neighbor_ids.union(set(visible_src_neighbors[neighbor_idx]))
                for node_id in src_unique_keys_temp:
                    if node_id in dst_id_neighbor_ids:
                        dst_mapping_dict[node_id] = num_bidirectional
            
            # Reconstruction for src (using only visible events)
            src_id_neighbor_ids = set()
            if len(src_unique_keys_temp) != 0 and len(visible_dst_node_ids) > 0:
                for node_id in src_unique_keys_temp:
                    if node_id in visible_dst_node_ids:
                        index = np.where(visible_dst_node_ids == node_id)
                        if len(index[0]) > 0:
                            neighbor_idx = index[0][0]  # Take first occurrence
                            src_id_neighbor_ids = src_id_neighbor_ids.union(set(visible_dst_neighbors[neighbor_idx]))
                for node_id in dst_unique_keys_temp:
                    if node_id in src_id_neighbor_ids:
                        src_mapping_dict[node_id] = num_bidirectional

            # Calculate appearances in each other's lists
            src_node_neighbor_counts_in_dst = torch.tensor([dst_mapping_dict.get(neighbor_id, 0) for neighbor_id in src_node_neighbor_ids]).float().to(self.device)
            dst_node_neighbor_counts_in_src = torch.tensor([src_mapping_dict.get(neighbor_id, 0) for neighbor_id in dst_node_neighbor_ids]).float().to(self.device)

            # Stack counts to get a two-column tensor for each node list
            src_nodes_appearances.append(torch.stack([torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device), src_node_neighbor_counts_in_dst], dim=1))
            dst_nodes_appearances.append(torch.stack([dst_node_neighbor_counts_in_src, torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)], dim=1))

        # Stack to form batch tensors
        src_nodes_appearances = torch.stack(src_nodes_appearances, dim=0)
        dst_nodes_appearances = torch.stack(dst_nodes_appearances, dim=0)

        # Restore original order if temporal masking was used
        if self.use_temporal_masking:
            # Create inverse mapping to restore original order
            restore_order = np.empty_like(time_sorted_indices)
            restore_order[time_sorted_indices] = np.arange(len(time_sorted_indices))
            src_nodes_appearances = src_nodes_appearances[restore_order]
            dst_nodes_appearances = dst_nodes_appearances[restore_order]

        return src_nodes_appearances, dst_nodes_appearances

    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, 
               src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray, 
               node_interact_times: np.ndarray = None, num_bidirectional: int = 2):
        """
        compute the BIE features of nodes in src_nodes_neighbor_ids and dst_nodes_neighbor_ids
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param src_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_nodes_neighbor_ids: ndarray, shape (batch_size, dst_max_seq_length)
        :param node_interact_times: ndarray, shape (batch_size, ) - needed for temporal masking
        :param num_bidirectional: int
        :return:
        """
        src_nodes_appearances, dst_nodes_appearances = self.count_nodes_appearances(
            src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
            src_nodes_neighbor_ids=src_nodes_neighbor_ids,
            dst_nodes_neighbor_ids=dst_nodes_neighbor_ids,
            node_interact_times=node_interact_times,
            num_bidirectional=num_bidirectional)

        src_nodes_bie_features = self.src_bie_encode_layer(src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_bie_features = self.dst_bie_encode_layer(dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        
        return src_nodes_bie_features, dst_nodes_bie_features


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor):
        """
        encode the inputs by Transformer encoder
        :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
        :return:
        """
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs)[0].transpose(0, 1)
        temp = self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return outputs