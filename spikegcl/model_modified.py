from spikegcl import neuron
import torch

from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import dropout_edge, mask_feature, to_torch_coo_tensor, degree
import torch.nn.functional as F


def creat_activation_layer(activation):
    if activation is None:
        return torch.nn.Identity()
    elif activation == "relu":
        return torch.nn.ReLU()
    elif activation == "elu":
        return torch.nn.ELU()
    else:
        raise ValueError("Unknown activation")


def creat_snn_layer(
    alpha=2.0,
    surrogate="sigmoid",
    v_threshold=5e-3,
    snn="PLIF",
):
    tau = 1.0

    if snn in ["LIF", "PLIF", "ALIF"]:
        return getattr(neuron, snn)(tau, alpha=alpha,
                                    surrogate=surrogate,
                                    v_threshold=v_threshold,
                                    detach=True,
                                    )
    elif snn == "IF":
        return neuron.IF(
            alpha=alpha, surrogate=surrogate, v_threshold=v_threshold, detach=True,
        )
    else:
        raise ValueError("Unknown SNN")

def create_das_snn_layer(
    alpha=2.0,
    surrogate="sigmoid",
    v_threshold=5e-3,
    channels=128,
    bins=20,
):
    return neuron.LAPLIF_deg_feat(v_threshold=v_threshold, ssize=channels, bins=bins)


class SpikeGCL(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        T: int = 32,
        alpha: float = 1.0,
        surrogate: str = "sigmoid",
        v_threshold: float = 5e-3,
        snn: str = "PLIF",
        reset: str = "zero",
        act: str = "elu",
        dropedge: float = 0.0,
        dropout: float = 0.5,
        shuffle: bool = True,
        bn: bool = True,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.snn = creat_snn_layer(
            alpha=alpha,
            surrogate=surrogate,
            v_threshold=v_threshold,
            snn=snn,
        )
        bn = torch.nn.BatchNorm1d if bn else torch.nn.Identity

        in_channels = [
            x.size(0) for x in torch.chunk(torch.ones(in_channels), T)
        ]
        for channel in in_channels:
            self.convs.append(GCNConv(channel, hidden_channels))
            self.bns.append(bn(channel))

        self.shared_bn = bn(hidden_channels)
        self.shared_conv = GCNConv(hidden_channels, hidden_channels)

        self.lin = torch.nn.Linear(hidden_channels, out_channels, bias=False)
        self.act = creat_activation_layer(act)
        self.drop_edge = dropedge
        self.T = T
        self.dropout = torch.nn.Dropout(dropout)
        self.reset = reset
        self.shuffle = shuffle

    def encode(self, x, edge_index, edge_weight=None):
        chunks = torch.chunk(x, self.T, dim=1)
        edge_index = to_torch_coo_tensor(edge_index, size=x.size(0))
        xs = []
        for i, x in enumerate(chunks):
            x = self.dropout(x)
            x = self.bns[i](x)
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            x = self.dropout(x)
            x = self.shared_bn(x)
            x = self.shared_conv(x, edge_index, edge_weight)
            x = self.snn(x)
            xs.append(x)
        self.snn.reset()
        return xs

    def decode(self, spikes):
        xs = []
        for spike in spikes:
            xs.append(self.lin(spike).sum(1))
        return xs

    def forward(self, x, edge_index, edge_weight=None):
        edge_index2, mask2 = dropout_edge(edge_index, p=self.drop_edge)

        if edge_weight is not None:
            edge_weight2 = edge_weight[mask2]
        else:
            edge_weight2 = None

        if self.shuffle:
            x2 = x[:, torch.randperm(x.size(1))]
        else:
            x2 = x

        s1 = self.encode(x, edge_index, edge_weight)
        s2 = self.encode(x2, edge_index2, edge_weight2)

        z1 = self.decode(s1)
        z2 = self.decode(s2)
        return z1, z2

    def loss(self, postive, negative, margin=0.0):
        loss = F.margin_ranking_loss(
            postive, negative, target=torch.ones_like(postive), margin=margin
        )
        return loss
    
class SpikeGCL_GraphClassification(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        T: int = 32,
        alpha: float = 1.0,
        surrogate: str = "sigmoid",
        v_threshold: float = 5e-3,
        snn: str = "PLIF",
        reset: str = "zero",
        act: str = "elu",
        dropedge: float = 0.0,
        dropout: float = 0.5,
        shuffle: bool = True,
        bn: bool = True,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.snn = creat_snn_layer(
            alpha=alpha,
            surrogate=surrogate,
            v_threshold=v_threshold,
            snn=snn,
        )
        bn = torch.nn.BatchNorm1d if bn else torch.nn.Identity

        in_channels = [
            x.size(0) for x in torch.chunk(torch.ones(in_channels), T)
        ]
        for channel in in_channels:
            self.convs.append(GCNConv(channel, hidden_channels))
            self.bns.append(bn(channel))

        self.shared_bn = bn(hidden_channels)
        self.shared_conv = GCNConv(hidden_channels, hidden_channels)

        self.lin = torch.nn.Linear(hidden_channels, out_channels, bias=False)
        self.act = creat_activation_layer(act)
        self.drop_edge = dropedge
        self.T = T
        self.dropout = torch.nn.Dropout(dropout)
        self.reset = reset
        self.shuffle = shuffle

    def encode(self, x, edge_index, edge_weight=None):
        chunks = torch.chunk(x, self.T, dim=1)
        edge_index = to_torch_coo_tensor(edge_index, size=x.size(0))
        xs = []
        for i, x in enumerate(chunks):
            x = self.dropout(x)
            x = self.bns[i](x)
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            x = self.dropout(x)
            x = self.shared_bn(x)
            x = self.shared_conv(x, edge_index, edge_weight)
            x = self.snn(x)
            xs.append(x)
        self.snn.reset()
        return xs

    def decode(self, spikes):
        xs = []
        for spike in spikes:
            xs.append(self.lin(spike).sum(1))
        return xs

    def forward(self, x, edge_index, edge_weight=None):
        edge_index2, mask2 = dropout_edge(edge_index, p=self.drop_edge)

        if edge_weight is not None:
            edge_weight2 = edge_weight[mask2]
        else:
            edge_weight2 = None

        if self.shuffle:
            x2 = x[:, torch.randperm(x.size(1))]
        else:
            x2 = x

        s1 = self.encode(x, edge_index, edge_weight)
        s2 = self.encode(x2, edge_index2, edge_weight2)

        z1 = self.decode(s1)
        z2 = self.decode(s2)
        return z1, z2

    def loss(self, postive, negative, margin=0.0):
        loss = F.margin_ranking_loss(
            postive, negative, target=torch.ones_like(postive), margin=margin
        )
        return loss
    
class SpikeGCL_DASGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        T: int = 32,
        alpha: float = 2.0,
        surrogate: str = "sigmoid",
        v_threshold: float = 5e-3,
        snn: str = "PLIF",
        reset: str = "zero",
        act: str = "elu",
        dropedge: float = 0.2,
        dropout: float = 0.5,
        shuffle: bool = True,
        bn: bool = True,
        deg_labels = None,
        bins = 20,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        # self.snn = creat_snn_layer(
        #     alpha=alpha,
        #     surrogate=surrogate,
        #     v_threshold=v_threshold,
        #     snn=snn,
        # )
        self.das_snn = create_das_snn_layer(
            alpha=alpha,
            surrogate=surrogate,
            v_threshold=v_threshold,
            channels=hidden_channels,
            bins=bins,
        )
        bn = torch.nn.BatchNorm1d if bn else torch.nn.Identity

        in_channels = [
            x.size(0) for x in torch.chunk(torch.ones(in_channels), T)
        ]
        for channel in in_channels:
            self.convs.append(GCNConv(channel, hidden_channels))
            self.bns.append(bn(channel))

        self.shared_bn = bn(hidden_channels)
        self.shared_conv = GCNConv(hidden_channels, hidden_channels)

        self.lin = torch.nn.Linear(hidden_channels, out_channels, bias=False)
        self.act = creat_activation_layer(act)
        self.drop_edge = dropedge
        self.T = T
        self.dropout = torch.nn.Dropout(dropout)
        self.reset = reset
        self.shuffle = shuffle
        self.deg_labels = deg_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, x, edge_index, edge_weight=None):
        chunks = torch.chunk(x, self.T, dim=1)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype = x.dtype)
        cur_degree = deg
        if self.deg_labels:
            cur_degree_np = cur_degree.cpu().numpy()
            # Map each degree in cur_degree to its corresponding label using the degree_to_label dictionary
            binned_degrees = torch.tensor([self.deg_labels[int(deg)] for deg in cur_degree_np], dtype=torch.long).to(self.device)

        edge_index = to_torch_coo_tensor(edge_index, size=x.size(0))
        xs = []
        for i, x in enumerate(chunks):
            x = self.dropout(x)
            x = self.bns[i](x)
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            x = self.dropout(x)
            x = self.shared_bn(x)
            x = self.shared_conv(x, edge_index, edge_weight)
            
            x = self.das_snn(x, binned_degrees)
            xs.append(x)
        self.das_snn.reset()
        return xs

    def decode(self, spikes):
        xs = []
        for spike in spikes:
            xs.append(self.lin(spike).sum(1))
        return xs

    def forward(self, x, edge_index, edge_weight=None):
        # edge_index2, mask2 = dropout_edge(edge_index, p=self.drop_edge)
        edge_index2, mask2 = dropout_edge(edge_index, p=0.0)

        if edge_weight is not None:
            edge_weight2 = edge_weight[mask2]
        else:
            edge_weight2 = None

        if self.shuffle:
            x2 = x[:, torch.randperm(x.size(1))]
        else:
            x2 = x

        s1 = self.encode(x, edge_index, edge_weight)
        s2 = self.encode(x2, edge_index2, edge_weight)

        z1 = self.decode(s1)
        z2 = self.decode(s2)
        return z1, z2

    def loss(self, postive, negative, margin=0.0):
        loss = F.margin_ranking_loss(
            postive, negative, target=torch.ones_like(postive), margin=margin
        )
        return loss
