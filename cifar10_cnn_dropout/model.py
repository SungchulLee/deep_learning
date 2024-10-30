import torch
import torch.nn as nn
import torch.nn.functional as F

from global_name_space import ARGS


class Net(nn.Module):
    def __init__(self, dropout_prob=ARGS.dropout_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.batchnorm1 = nn.BatchNorm2d(6)  # BatchNorm after the first convolution
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchnorm2 = nn.BatchNorm2d(16)  # BatchNorm after the second convolution
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        if ARGS.batchnorm:
            x = self.conv1(x)
            x = self.batchnorm1(x) 
            x = self.pool(F.relu(x))
            x = self.conv2(x)
            x = self.batchnorm2(x) 
            x = self.pool(F.relu(x))
        else:
            x = self.conv1(x)
            #x = self.batchnorm1(x) 
            x = self.pool(F.relu(x))
            x = self.conv2(x)
            #x = self.batchnorm2(x) 
            x = self.pool(F.relu(x))
        x = torch.flatten(x, 1)
        if ARGS.dropout:
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
        else:
            x = F.relu(self.fc1(x))
            #x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            #x = self.dropout2(x)
        x = self.fc3(x)
        return x