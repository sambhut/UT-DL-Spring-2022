import torch

class ValueNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 128)
        self.fc3 = torch.nn.Linear(128, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
