import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='linear_regression_gradient_descent_numpy')
parser.add_argument('--epochs', type=int, default=1_000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--seed', type=int, default=1)
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)

ARGS.path = './save/saved_object.pkl'
os.makedirs("./save",exist_ok=True)