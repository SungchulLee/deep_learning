import argparse
import numpy as np
import os
import torch

parser = argparse.ArgumentParser(description='linear_regression_gradient_descent_torch')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)

ARGS.train_kwargs = {'batch_size': ARGS.batch_size}
ARGS.test_kwargs = {'batch_size': ARGS.batch_size}

ARGS.path_w = './model/w.pth'
ARGS.path_b = './model/b.pth'
os.makedirs("./model",exist_ok=True)