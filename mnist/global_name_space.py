import argparse
import numpy as np
import os
import torch

parser = argparse.ArgumentParser(description='mnist')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR',
                    help='momentum (default: 0.9)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--batchnorm', type=bool, default=False, metavar='S',
                    help='batchnorm activation (default: False)')
parser.add_argument('--dropout', type=bool, default=False, metavar='S',
                    help='dropout activation (default: False)')
parser.add_argument('--dropout_prob_1', type=float, default=0.25, metavar='S',
                    help='dropout probability (default: 0.5)')
parser.add_argument('--dropout_prob_2', type=float, default=0.5, metavar='S',
                    help='dropout probability (default: 0.5)')
parser.add_argument('--scheduler', type=bool, default=True, metavar='M',
                    help='scheduler activation (default: False)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='ables CUDA training (default: False)')
parser.add_argument('--mps', action='store_true', default=True,
                    help='ables macOS GPU training (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
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

ARGS.train_kwargs = {'batch_size': ARGS.batch_size}
ARGS.test_kwargs = {'batch_size': ARGS.test_batch_size}
if ARGS.use_cuda:
    cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    ARGS.train_kwargs.update(cuda_kwargs)
    ARGS.test_kwargs.update(cuda_kwargs)

ARGS.classes = ("0","1","2","3","4","5","6","7","8","9")
ARGS.path = './model/model.pth'
os.makedirs("./model",exist_ok=True)