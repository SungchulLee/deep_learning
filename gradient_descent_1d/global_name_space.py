import argparse

parser = argparse.ArgumentParser(description='mnist')
parser.add_argument('--lr', type=float, default=2e-2, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=int(1e2), metavar='N',
                    help='number of epochs to train (default: 2)')
ARGS = parser.parse_args()

ARGS.compute_loss = lambda theta : theta**2
ARGS.compute_gradient = lambda theta : 2*theta
ARGS.apply_gradient_descent = lambda theta, grad, lr: theta - lr * grad