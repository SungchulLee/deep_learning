import torch
import torch.nn as nn
import torch.nn.functional as F

from global_name_space import ARGS


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)  
        self.dropout1 = nn.Dropout(ARGS.dropout_prob_1)
        self.dropout2 = nn.Dropout(ARGS.dropout_prob_2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x): # x.shape = (64,1,28,28)
        if ARGS.batchnorm:
            x = self.conv1(x)
            # (64,32,26,26) = self.conv1( (64,1,28,28) ) 
            x = self.batchnorm1(x)
            # (64,32,26,26) = self.batchnorm1( (64,32,26,26) ) 
            x = F.relu(x)
            # (64,32,26,26) = F.relu( (64,32,26,26) ) 
            x = self.conv2(x)
            # (64,64,24,24) = self.conv2( (64,32,26,26) ) 
            x = self.batchnorm2(x)
            # (64,64,24,24) = batchnorm2( (64,64,24,24) ) 
            x = F.relu(x)
            # (64,64,24,24) = F.relu( (64,64,24,24) ) 
            x = self.pool(x)
            # (64,64,12,12) = self.pool( (64,64,24,24) ) 
        else:
            x = self.conv1(x)
            # (64,32,26,26) = self.conv1( (64,1,28,28) ) 
            # x = self.batchnorm1(x)
            # (64,32,26,26) = self.batchnorm1( (64,32,26,26) ) 
            x = F.relu(x)
            # (64,32,26,26) = F.relu( (64,32,26,26) ) 
            x = self.conv2(x)
            # (64,64,24,24) = self.conv2( (64,32,26,26) ) 
            # x = self.batchnorm2(x)
            # (64,64,24,24) = batchnorm2( (64,64,24,24) ) 
            x = F.relu(x)
            # (64,64,24,24) = F.relu( (64,64,24,24) ) 
            x = self.pool(x)
            # (64,64,12,12) = self.pool( (64,64,24,24) ) 
        x = torch.flatten(x, 1)
        # (64,9216) = torch.flatten( (64,64,12,12), 1 ) 
        if ARGS.dropout:
            x = self.dropout1(x)
            # (64,9216) = self.dropout1( (64,9216) ) 
            x = self.fc1(x)
            # (64,128) = self.fc1( (64,9216) ) 
            x = F.relu(x)
            # (64,128) = F.relu( (64,128) ) 
            x = self.dropout2(x)
            # (64,128) = self.dropout2( (64,128) ) 
            x = self.fc2(x)
            # (64,10) = self.fc2( (64,128) ) 
        else:
            # x = self.dropout1(x)
            # (64,9216) = self.dropout1( (64,9216) ) 
            x = self.fc1(x)
            # (64,128) = self.fc1( (64,9216) ) 
            x = F.relu(x)
            # (64,128) = F.relu( (64,128) ) 
            # x = self.dropout2(x)
            # (64,128) = self.dropout2( (64,128) ) 
            x = self.fc2(x)
            # (64,10) = self.fc2( (64,128) ) 
        return x