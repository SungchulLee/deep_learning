import argparse
import numpy as np
import os
import torch

parser = argparse.ArgumentParser(description='cifar10_cnn')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR',
                    help='momentum (default: 0.9)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--batchnorm', type=bool, default=False, metavar='S',
                    help='batchnorm activation (default: False)')
parser.add_argument('--dropout', type=bool, default=False, metavar='S',
                    help='dropout activation (default: False)')
parser.add_argument('--dropout-prob', type=float, default=0.5, metavar='S',
                    help='dropout probability (default: 0.5)')
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
# elif ARGS.use_mps:
#     ARGS.device = torch.device("mps")
else:
    ARGS.device = torch.device("cpu")

ARGS.train_kwargs = {'batch_size': ARGS.batch_size, 'shuffle': True, "num_workers": 2}
ARGS.test_kwargs = {'batch_size': ARGS.test_batch_size, 'shuffle': False, "num_workers": 2}
if ARGS.use_cuda:
    cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    ARGS.train_kwargs.update(cuda_kwargs)
    ARGS.test_kwargs.update(cuda_kwargs)

ARGS.classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
ARGS.path = './model/model.pth'
os.makedirs("./model",exist_ok=True)
