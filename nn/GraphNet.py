import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GraphNet(torch.nn.Module):
    def __init__(self, n_features, n_actions, emb_size=32, edge_index=None):
        super(GraphNet, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.edge_index = edge_index.to(self.device)
        self.conv1 = GCNConv(n_features, emb_size)
        self.conv2 = GCNConv(emb_size, emb_size)
        self.fc = Linear(emb_size, n_actions)

        self.opt = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        self.loss = torch.nn.MSELoss()
        self.to(self.device)

    def set_edge_index(self, edge_index):
        self.edge_index = edge_index

    def forward(self, x):
        assert self.edge_index is not None, "Edge indices have not been set, use 'set_edge_index(edge_index)'"
        # data = Data(x=x, edge_index=self.edge_index)
        x = self.conv1(x, self.edge_index)
        x = x.relu()
        x = self.fc(self.conv2(x, self.edge_index))
        return x

    def save_model(self, path):
        torch.save(self.cpu().state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
