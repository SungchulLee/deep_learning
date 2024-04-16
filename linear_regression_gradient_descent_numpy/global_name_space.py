import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='linear_regression_gradient_descent_numpy')
parser.add_argument('--epochs', type=int, default=1_000, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)

ARGS.path = './save/saved_object.pkl'
os.makedirs("./save",exist_ok=True)