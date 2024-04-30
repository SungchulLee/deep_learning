import numpy as np
import torch
from torch.utils import data

from global_name_space import ARGS


def load_data():
    x_train = np.random.uniform(size=(ARGS.batch_size,1))
    y_train = 1 + 2*x_train + np.random.normal(scale=0.1,size=(ARGS.batch_size,1))

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    train_ds = data.TensorDataset(x_train, y_train)

    trainloader = data.DataLoader(train_ds, **ARGS.train_kwargs)

    return trainloader