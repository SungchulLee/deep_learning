import torch
import numpy as np
from torch.utils import data 

from global_name_space import ARGS


def load_data():
    x = np.random.uniform(size=(1000,1))
    y = 1 + 2*x + np.random.normal(scale=0.1,size=(1000,1))

    x_train = torch.tensor(x, dtype=torch.float32) # (100, 1)
    y_train = torch.tensor(y, dtype=torch.float32) # (100, 1)

    train_ds = data.TensorDataset(x_train, y_train)

    trainloader = data.DataLoader(train_ds, **ARGS.train_kwargs)

    x_ = np.random.uniform(size=(1000,1))
    y_ = 1 + 2*x_ + np.random.normal(scale=0.1,size=(1000,1))

    x_test = torch.tensor(x_, dtype=torch.float32) # (100, 1)
    y_test = torch.tensor(y_, dtype=torch.float32) # (100, 1)

    test_ds = data.TensorDataset(x_test, y_test)

    testloader = data.DataLoader(test_ds, **ARGS.test_kwargs)
    
    return trainloader, testloader