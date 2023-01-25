import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

from dig.xgraph.models import GNNBasic


class GCN(GNNBasic):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int=256,
        dropout: float=0.5
    ) -> None:
        super(GCN, self).__init__()

        self.convs = nn.ModuleList(
            [
                gnn.GCNConv(input_dim, hidden_dim),
                gnn.GCNConv(hidden_dim, hidden_dim),
                gnn.GCNConv(hidden_dim, hidden_dim)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        x = self.get_emb(x, edge_index, batch)
        return self.classifier(gnn.global_max_pool(x, batch))

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, _ = self.arguments_read(*args, **kwargs)
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return x

    def predict(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        return self(x, edge_index, batch).argmax(dim=1).squeeze()
