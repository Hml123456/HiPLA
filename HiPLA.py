import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DenseGCNConv, GINConv, global_mean_pool
from torch_geometric.nn.conv import MessagePassing


def _rbf(D, D_min=0., D_max=6., D_count=9, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    return RBF


def gnn_norm(x, norm):
    batch_size, num_nodes, num_channels = x.size()
    x = x.view(-1, num_channels)
    x = norm(x)
    x = x.view(batch_size, num_nodes, num_channels)

    return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.mlp(x)


class FC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, drop_rate, output_dim):
        super(FC, self).__init__()

        self.predict = nn.ModuleList()
        self.predict.append(MLP(input_dim, hidden_dim, drop_rate))
        for _ in range(num_layer - 2):
            self.predict.append(MLP(hidden_dim, hidden_dim, drop_rate))
        self.predict.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, h):

        for layer in self.predict:
            h = layer(h)

        return h


class HIL(MessagePassing):
    def __init__(self, input_dim, output_dim, drop_rate, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HIL, self).__init__(**kwargs)

        self.mlp_coord = MLP(9, input_dim, 0.0)
        self.out = MLP(input_dim, output_dim, drop_rate)

    def message(self, x_j, x_i, radial, index):
        return x_j * radial

    def forward(self, x, data, edge_index):
        res = x

        pos, size = data.pos, None
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        dist = torch.norm(coord_diff, p=2, dim=-1)
        radial = self.mlp_coord(_rbf(dist, device=x.device))
        x = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)

        x = self.out(x) + res

        return x


class GIGNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(GIGNBlock, self).__init__()

        self.gconv_intra = HIL(input_dim, output_dim, drop_rate)
        self.gconv_inter = HIL(input_dim, output_dim, drop_rate)

    def forward(self, x, data):
        x_intra = self.gconv_intra(x, data, data.edge_index_intra)
        x_inter = self.gconv_inter(x, data, data.edge_index_inter)
        x = (x_intra + x_inter) / 2

        return x


class GINLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim, drop_rate):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gin_conv = GINConv(self.mlp)

    def forward(self, x, edge_index):
        return self.gin_conv(x, edge_index)


class GlobalGINEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=3, drop_rate=0.1):
        super().__init__()
        self.num_layers = num_layers

        self.gin_layers = nn.ModuleList()

        self.gin_layers.append(GINLayer(input_dim, hidden_dim, drop_rate))

        for _ in range(num_layers - 1):
            self.gin_layers.append(GINLayer(hidden_dim, hidden_dim, drop_rate))


        self.global_mlp = MLP(hidden_dim, hidden_dim, drop_rate)

    def forward(self, x, edge_index, batch):

        for gin_layer in self.gin_layers:
            x = gin_layer(x, edge_index)
            x = F.relu(x)


        global_repr = global_mean_pool(x, batch)


        global_repr = self.global_mlp(global_repr)

        return global_repr


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, heads, drop_rate):
        super().__init__()

        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = MLP(hidden_dim, hidden_dim, drop_rate)

    def forward(self, q, k, v):
        res = q.sum(dim=1)

        batch_size, seqlen_q, _ = q.shape
        _, seqlen_k, _ = k.shape

        Q = self.W_Q(q)  # [batch_size, seqlen_q, hidden_dim]
        K = self.W_K(k)
        V = self.W_V(v)

        Q = Q.view(batch_size, seqlen_q, self.heads, self.head_dim).transpose(1,
                                                                              2)  # [batch_size, heads, seqlen_q, head_dim]
        K = K.view(batch_size, seqlen_k, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seqlen_k, self.heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.head_dim)  # [batch_size, num_heads, seqlen_q, seqlen_k]
        attention = torch.softmax(energy, dim=-1)  # [batch_size, num_heads, seqlen_q, seqlen_k]
        x = torch.matmul(attention, V)  # [batch_size, num_heads, seqlen_q, head_dim]
        x = x.transpose(1, 2).contiguous().view(batch_size, seqlen_q,
                                                self.hidden_dim)  # [batch_size, seqlen_q, hidden_dim]
        x = x.sum(dim=1)

        x = self.W_O(x) + res

        return x, attention


class HiPLA(nn.Module):

    def __init__(self, node_dim, hidden_dim, heads=1, drop_rate=0.1):
        super().__init__()

        self.embedding = MLP(node_dim, hidden_dim, 0.0)
        self.GIGNBlock1 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.GIGNBlock2 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.GIGNBlock3 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.GIGNBlock4 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.GIGNBlock5 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)


        self.gin_encoder_lig = GlobalGINEncoder(hidden_dim, hidden_dim, num_layers=3, drop_rate=drop_rate)
        self.gin_encoder_pro = GlobalGINEncoder(hidden_dim, hidden_dim, num_layers=3, drop_rate=drop_rate)

        self.attblock1 = AttentionBlock(hidden_dim, heads, drop_rate)
        self.attblock2 = AttentionBlock(hidden_dim, heads, drop_rate)


        gate_input_dim = hidden_dim * 4
        self.gate_layer = nn.Linear(gate_input_dim, hidden_dim)
        self.fc = FC(hidden_dim, hidden_dim, 2, drop_rate, 1)

    def make_edge_index(self, data):

        data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
        data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]

    def remap_edge_index(self, edge_index, node_mask):

        if edge_index.size(1) == 0:
            return edge_index


        global_to_local = torch.full((node_mask.size(0),), -1, dtype=torch.long, device=edge_index.device)
        local_nodes = torch.where(node_mask)[0]
        global_to_local[local_nodes] = torch.arange(local_nodes.size(0), device=edge_index.device)

        remapped_edge_index = global_to_local[edge_index]

        valid_edges = (remapped_edge_index[0] >= 0) & (remapped_edge_index[1] >= 0)

        return remapped_edge_index[:, valid_edges]

    def forward(self, data):

        x = self.embedding(data.x)


        self.make_edge_index(data)
        x = self.GIGNBlock1(x, data)
        x = self.GIGNBlock2(x, data)
        x = self.GIGNBlock3(x, data)
        x = self.GIGNBlock4(x, data)
        x = self.GIGNBlock5(x, data)


        batch = data.batch
        split = data.split
        lig_mask = (split == 0)
        pro_mask = (split == 1)

        x_lig = x[lig_mask]
        x_pro = x[pro_mask]
        batch_lig = batch[lig_mask]
        batch_pro = batch[pro_mask]


        edge_index_lig_local = self.remap_edge_index(data.edge_index_intra_lig, lig_mask)
        edge_index_pro_local = self.remap_edge_index(data.edge_index_intra_pro, pro_mask)

        global_lig_gin = self.gin_encoder_lig(x_lig, edge_index_lig_local, batch_lig)
        global_pro_gin = self.gin_encoder_pro(x_pro, edge_index_pro_local, batch_pro)


        x_lig, _ = to_dense_batch(x_lig, batch_lig)
        x_pro, _ = to_dense_batch(x_pro, batch_pro)


        l2p, _ = self.attblock1(x_lig, x_pro, x_pro)
        p2l, _ = self.attblock2(x_pro, x_lig, x_lig)


        gate_input = torch.cat([l2p, p2l, global_lig_gin, global_pro_gin], dim=-1)

        gate = torch.sigmoid(self.gate_layer(gate_input))
        x = gate * l2p + (1 - gate) * p2l


        x = self.fc(x)

        return x.view(-1)