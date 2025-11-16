import torch
import numpy as np
import torch.nn as nn
import math
from utils.utils import NeighborSampler
from models.modules import TimeEncoder

class RandomProjectionModule(nn.Module):
    def __init__(self, node_num: int, edge_num: int, dim_factor: int, num_layer: int, time_decay_weight: list[float],
                 device: str, use_matrix: bool, beginning_time: np.float64, not_scale: bool, enforce_dim: int):

        super(RandomProjectionModule, self).__init__()

        self.node_num = node_num
        self.edge_num = edge_num
        if enforce_dim != -1:
            self.dim = enforce_dim
        else:
            self.dim = min(int(math.log(self.edge_num * 2)) * dim_factor, node_num)
        self.num_layer = num_layer
        self.time_decay_weight = time_decay_weight
        self.device = device
        self.use_matrix = use_matrix
        self.node_feature_dim = 128
        self.not_scale = not_scale
        
        self.begging_time = nn.Parameter(torch.tensor(beginning_time), requires_grad=False)
        self.now_time = nn.Parameter(torch.tensor(beginning_time), requires_grad=False)
        
        # LAMBDAS
        self.lambdas = time_decay_weight
        self.M = len(self.lambdas)

        
        self.random_projections_multi = nn.ModuleList()
        for _ in range(self.M):
            pl = nn.ParameterList()
            for i in range(self.num_layer + 1):
                if i == 0:
                    pl.append(nn.Parameter(torch.normal(0, 1 / math.sqrt(self.dim), (self.node_num, self.dim)), requires_grad=False)) #init con matrice P 
                else:
                    pl.append(nn.Parameter(torch.zeros_like(pl[i - 1]), requires_grad=False))
            self.random_projections_multi.append(pl)
    
        self.pair_wise_feature_dim = (2 * self.num_layer + 2) ** 2
        self.mlp = nn.Sequential(
            nn.Linear(self.pair_wise_feature_dim, self.pair_wise_feature_dim * 4),
            nn.ReLU(),
            nn.Linear(self.pair_wise_feature_dim * 4, self.pair_wise_feature_dim)
        )

        # Init dei pesi
        self.lambda_weights = nn.Parameter(torch.zeros(self.num_layer + 1, self.M))


    def update(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        src_node_ids = torch.from_numpy(src_node_ids).to(self.device)
        dst_node_ids = torch.from_numpy(dst_node_ids).to(self.device)
        next_time = node_interact_times[-1]
        node_interact_times = torch.from_numpy(node_interact_times).to(dtype=torch.float, device=self.device)

        for m, lam in enumerate(self.lambdas):
            time_weight = torch.exp(-lam * (next_time - node_interact_times))[:, None]

            #Mantiene direttamente le F
            for i in range(1, self.num_layer + 1):
                factor = np.power(np.exp(-lam * (next_time - self.now_time.cpu().numpy())), i)
                self.random_projections_multi[m][i].data = self.random_projections_multi[m][i].data * factor

            # agg
            for i in range(self.num_layer, 0, -1):
                src_msg = self.random_projections_multi[m][i - 1][dst_node_ids] * time_weight
                dst_msg = self.random_projections_multi[m][i - 1][src_node_ids] * time_weight
                self.random_projections_multi[m][i].scatter_add_(dim=0, index=src_node_ids[:, None].expand(-1, self.dim), src=src_msg)
                self.random_projections_multi[m][i].scatter_add_(dim=0, index=dst_node_ids[:, None].expand(-1, self.dim), src=dst_msg)

        self.now_time.data = torch.tensor(next_time, device=self.device)


    def get_random_projections(self, node_ids: np.ndarray):
        k1 = self.num_layer + 1

        # Raccolta
        per_scale = []
        for m in range(self.M):
            stack_m = torch.stack(
                [self.random_projections_multi[m][i][node_ids, :] for i in range(k1)],
                dim=1
            )
            per_scale.append(stack_m)

        S = torch.stack(per_scale, dim=0) 

        #Media pesata
        W = torch.softmax(self.lambda_weights, dim=1)
        Wb = W.t().contiguous().view(self.M, 1, k1, 1) 
        fused = (S * Wb).sum(dim=0) 

        return [fused[:, i, :] for i in range(k1)]

    #OG da qui in poi

    def get_pair_wise_feature(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray):
        src_random_projections = torch.stack(self.get_random_projections(src_node_ids), dim=1)
        dst_random_projections = torch.stack(self.get_random_projections(dst_node_ids), dim=1)
        random_projections = torch.cat([src_random_projections, dst_random_projections], dim=1)
        random_feature = torch.matmul(random_projections, random_projections.transpose(1, 2)).reshape(len(src_node_ids), -1)
        if self.not_scale:
            return self.mlp(random_feature)
        else:
            random_feature[random_feature < 0] = 0
            random_feature = torch.log(random_feature + 1.0)
            return self.mlp(random_feature)

    def reset_random_projections(self):
        for m in range(self.M):
            for i in range(1, self.num_layer + 1):
                nn.init.zeros_(self.random_projections_multi[m][i])
            if not self.use_matrix:
                nn.init.normal_(self.random_projections_multi[m][0], mean=0, std=1 / math.sqrt(self.dim))
        self.now_time.data = self.begging_time.clone()

    def backup_random_projections(self):
        return self.now_time.clone(), [
            [self.random_projections_multi[m][i].clone() for i in range(1, self.num_layer + 1)] for m in range(self.M)
        ]

    def reload_random_projections(self, random_projections):
        now_time, payload = random_projections
        self.now_time.data = now_time.clone()
        for m in range(self.M):
            for i in range(1, self.num_layer + 1):
                self.random_projections_multi[m][i].data = payload[m][i - 1].clone()


#class TPNet_Weighted(torch.nn.Module):
class TPNet(torch.nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, dropout: float, random_projections: RandomProjectionModule,
                 num_layers: int, num_neighbors: int, device: str):
        #super(TPNet_Weighted, self).__init__()
        super(TPNet, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.dropout = dropout
        self.device = device
        self.num_nodes = self.node_raw_features.shape[0]

        self.random_projections = random_projections
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.embedding_module = TPNetEmbedding(
            node_raw_features=self.node_raw_features,
            edge_raw_features=self.edge_raw_features,
            neighbor_sampler=neighbor_sampler,
            time_encoder=self.time_encoder,
            node_feat_dim=self.node_feat_dim,
            edge_feat_dim=self.edge_feat_dim,
            time_feat_dim=self.time_feat_dim,
            num_layers=num_layers,
            num_neighbors=num_neighbors,
            dropout=self.dropout,
            random_projections=self.random_projections)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        node_embeddings = self.embedding_module.compute_node_temporal_embeddings(
            node_ids=np.concatenate([src_node_ids, dst_node_ids]),
            src_node_ids=np.tile(src_node_ids, 2),
            dst_node_ids=np.tile(dst_node_ids, 2),
            node_interact_times=np.tile(node_interact_times, 2))
        src_node_embeddings = node_embeddings[:len(src_node_ids)]
        dst_node_embeddings = node_embeddings[len(src_node_ids):]
        return src_node_embeddings, dst_node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.embedding_module.neighbor_sampler = neighbor_sampler
        if self.embedding_module.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.embedding_module.neighbor_sampler.seed is not None
            self.embedding_module.neighbor_sampler.reset_random_state()

class TPNetEmbedding(nn.Module):
    def __init__(self, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor, neighbor_sampler: NeighborSampler,
                 time_encoder: nn.Module, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_layers: int, num_neighbors: int, dropout: float, random_projections: RandomProjectionModule):
        super(TPNetEmbedding, self).__init__()
        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.neighbor_sampler = neighbor_sampler
        self.time_encoder = time_encoder
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.random_projections = random_projections

        if self.random_projections is None:
            self.random_feature_dim = 0
        else:
            self.random_feature_dim = self.random_projections.pair_wise_feature_dim * 2

        self.projection_layer = nn.Sequential(
            nn.Linear(node_feat_dim + edge_feat_dim + time_feat_dim + self.random_feature_dim, node_feat_dim * 2),
            nn.ReLU(),
            nn.Linear(node_feat_dim * 2, node_feat_dim))

        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=self.num_neighbors, num_channels=self.node_feat_dim, token_dim_expansion_factor=0.5,
                     channel_dim_expansion_factor=4.0, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        device = self.node_raw_features.device
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=self.num_neighbors)

        neighbor_node_features = self.node_raw_features[torch.from_numpy(neighbor_node_ids)]
        neighbor_delta_times = torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(device)
        neighbor_delta_times = torch.log(neighbor_delta_times + 1.0)
        neighbor_time_features = self.time_encoder(neighbor_delta_times)
        neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]

        if self.random_projections is not None:
            concat_neighbor_random_features = self.random_projections.get_pair_wise_feature(
                src_node_ids=np.tile(neighbor_node_ids.reshape(-1), 2),
                dst_node_ids=np.concatenate([np.repeat(src_node_ids, self.num_neighbors), np.repeat(dst_node_ids, self.num_neighbors)]))
            neighbor_random_features = torch.cat(
                [concat_neighbor_random_features[:len(node_ids) * self.num_neighbors], concat_neighbor_random_features[len(node_ids) * self.num_neighbors:]], dim=1).reshape(len(node_ids), self.num_neighbors, -1)
            neighbor_combine_features = torch.cat(
                [neighbor_node_features, neighbor_time_features, neighbor_edge_features, neighbor_random_features], dim=2)
        else:
            neighbor_combine_features = torch.cat(
                [neighbor_node_features, neighbor_time_features, neighbor_edge_features], dim=2)

        embeddings = self.projection_layer(neighbor_combine_features)
        embeddings.masked_fill(torch.from_numpy(neighbor_node_ids == 0)[:, :, None].to(device), 0)
        for mlp_mixer in self.mlp_mixers:
            embeddings = mlp_mixer(embeddings)
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        super(FeedForwardNet, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, int(dim_expansion_factor * input_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim_expansion_factor * input_dim), input_dim),
            nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        return self.ffn(x)

class MLPMixer(nn.Module):
    def __init__(self, num_tokens: int, num_channels: int, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.0):
        super(MLPMixer, self).__init__()
        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(num_tokens, token_dim_expansion_factor, dropout)
        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(num_channels, channel_dim_expansion_factor, dropout)

    def forward(self, input_tensor: torch.Tensor):
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        output_tensor = hidden_tensor + input_tensor
        hidden_tensor = self.channel_norm(output_tensor)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        return hidden_tensor + output_tensor
