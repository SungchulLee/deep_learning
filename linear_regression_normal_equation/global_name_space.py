import argparse
import numpy as np

parser = argparse.ArgumentParser(description='linear_regression_normal_equation')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)