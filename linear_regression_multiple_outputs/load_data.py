import numpy as np
import torch
from torch.utils import data

from global_name_space import ARGS


def load_data():
    # Define NumPy arrays for inputs and targets
    x = np.array([[73, 67, 43],
                  [91, 88, 64],
                  [87, 134, 58],
                  [102, 43, 37],
                  [69, 96, 70]], dtype=np.float32) # (5, 3)
    y = np.array([[56, 70],
                  [81, 101],
                  [119, 133],
                  [22, 37],
                  [103, 119]], dtype=np.float32) # (5, 2)

    # Convert NumPy arrays to PyTorch tensors using torch.from_numpy
    # Converts the NumPy array inputs to a PyTorch tensor.
    # This function torch.from_numpy creates a PyTorch tensor
    # that shares the same underlying data as the NumPy array.
    # It's important to note that torch.from_numpy creates a PyTorch tensor
    # that shares the memory with the original NumPy array.
    # This means that modifications to the tensor will be reflected
    # in the original NumPy array and vice versa.
    # This behavior is beneficial for memory efficiency
    # since it avoids unnecessary data copying.
    x_train = torch.from_numpy(x).to(torch.float32)  # (100, 1)
    y_train = torch.from_numpy(y).to(torch.float32)  # (100, 1)

    train_ds = data.TensorDataset(x_train, y_train)

    trainloader = data.DataLoader(train_ds, **ARGS.train_kwargs)

    return trainloader