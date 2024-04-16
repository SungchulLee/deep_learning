import argparse
import numpy as np

parser = argparse.ArgumentParser(description='mnist')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=2_000, metavar='N',
                    help='number of epochs to train (default: 2)')
ARGS = parser.parse_args()

ARGS.compute_loss = lambda x, y : np.sin(x)**10 + np.cos(10+x*y)*np.cos(x)
ARGS.compute_gradient = lambda x, y : (
    np.sin(x)**9*np.cos(x) - y*np.sin(10+x*y)*np.cos(x) - np.cos(10+x*y)*np.sin(x),
    - x*np.sin(10+x*y)*np.cos(x)
)
ARGS.apply_gradient_descent = lambda theta, grad, lr: [theta_ - lr * grad_ for theta_, grad_ in zip(theta, grad)]