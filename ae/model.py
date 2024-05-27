import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        return self.fc2(F.relu(self.fc1(x)))
    
    def decode(self, z):
        return torch.sigmoid(self.fc4(F.relu(self.fc3(z))))

    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        return self.decode(z)