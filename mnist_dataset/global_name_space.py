import argparse
import numpy as np
import os
import torch

parser = argparse.ArgumentParser(description='mnist_softmax_regression')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR',
                    help='momentum (default: 0.9)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--input_size', type=int, default=784, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--batchnorm', type=bool, default=False, metavar='S',
                    help='batchnorm activation (default: False)')
parser.add_argument('--dropout', type=bool, default=False, metavar='S',
                    help='dropout activation (default: False)')
parser.add_argument('--dropout-prob', type=float, default=0.5, metavar='S',
                    help='dropout probability (default: 0.5)')
parser.add_argument('--scheduler', type=bool, default=False, metavar='M',
                    help='scheduler activation (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='ables CUDA training (default: False)')
parser.add_argument('--mps', action='store_true', default=True,
                    help='ables macOS GPU training (default: False)')
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)

ARGS.use_cuda = ARGS.cuda and torch.cuda.is_available()
ARGS.use_mps = ARGS.mps and torch.backends.mps.is_available()
if ARGS.use_cuda:
    ARGS.device = torch.device("cuda")
elif ARGS.use_mps:
    ARGS.device = torch.device("mps")
else:
    ARGS.device = torch.device("cpu")

ARGS.train_kwargs = {'batch_size': ARGS.batch_size}
ARGS.test_kwargs = {'batch_size': ARGS.test_batch_size}
if ARGS.use_cuda:
    cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    ARGS.train_kwargs.update(cuda_kwargs)
    ARGS.test_kwargs.update(cuda_kwargs)

ARGS.classes = ("0","1","2","3","4","5","6","7","8","9")
ARGS.path = './model/model.pth'
os.makedirs("./model",exist_ok=True)

"""
What is num_workers in the code?

In the context of PyTorch's `DataLoader`, 
the `num_workers` parameter specifies the number of subprocesses to use for data loading. 
It determines how many parallel subprocesses will be used to load the data in the background 
while the model is training.

Setting a higher value for `num_workers` allows for more efficient data loading, 
especially when you are working with large datasets. 
Each worker operates independently, loading a batch of data in parallel, 
which can significantly speed up the data loading process. 
However, setting `num_workers` to a very high value might lead to excessive use of system resources 
and could cause issues. 
It's often a balance between achieving faster data loading and not overloading the system.

For example, in the code you provided:

```python
trainloader = data.DataLoader(trainset, **ARGS.train_kwargs)
```

If `ARGS.train_kwargs` contains a key-value pair like `num_workers=4`, 
it means that four subprocesses will be used for loading training data in parallel. 
Adjusting the value of `num_workers` based on your system's capabilities can help optimize 
the data loading performance during training.
"""