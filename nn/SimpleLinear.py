import torch
from torch.nn import Linear


class SimpleLinear(torch.nn.Module):
    def __init__(self, n_features, n_actions):
        super(SimpleLinear, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.fc = Linear(n_features, n_actions)
        self.opt = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        self.loss = torch.nn.MSELoss()
        self.to(self.device)

    def forward(self, x):
        return self.fc(x)

    def save_model(self, path):
        torch.save(self.cpu().state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
