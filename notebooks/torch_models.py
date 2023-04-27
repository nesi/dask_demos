from torch import nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.dense0 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X


# adapted from https://github.com/skorch-dev/skorch/blob/master/notebooks/MNIST.ipynb
class SimpleCNN(nn.Module):
    def __init__(self, input_dims, n_chans, hidden_dim, output_dim, dropout):
        super().__init__()
        self.input_dims = input_dims
        self.conv1 = nn.Conv2d(1, n_chans, kernel_size=3)
        self.conv2 = nn.Conv2d(n_chans, n_chans * 2, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=dropout)

        # number channels * width * height
        # TODO compute width/height from input_dims
        self.fc1 = nn.Linear(n_chans * 2 * 5 * 5, hidden_dim)

        self.fc1_drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = X.reshape((-1, 1) + tuple(self.input_dims))
        X = F.relu(F.max_pool2d(self.conv1(X), 2))
        X = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(X)), 2))
        X = X.view(-1, X.size(1) * X.size(2) * X.size(3))
        X = F.relu(self.fc1_drop(self.fc1(X)))
        X = F.softmax(self.fc2(X), dim=-1)
        return X
