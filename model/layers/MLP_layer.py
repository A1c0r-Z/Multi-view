from torch import nn


class MLPLayer(nn.Module):
    def __init__(self, d_model, out, hidden, drop_prob=0.1):
        super(MLPLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, out)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x
