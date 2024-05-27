import argparse
import os
import torch

parser = argparse.ArgumentParser(description='vae')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N', \
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100, metavar='N', \
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', \
                    help='input batch size for training (default: 128)')
parser.add_argument('--path', type=str, default="./model", metavar='N', \
                    help='path to store model.state_dict()')
parser.add_argument('--path_reconstructed_images', type=str, default="./reconstructed_images", metavar='N', \
                    help='path to store reconstructed images')
parser.add_argument('--path_generated_images', type=str, default="./generated_images", metavar='N', \
                    help='path to store generated images')
parser.add_argument('cuda', action='store_true', default=True, \
                    help='disables CUDA training')
parser.add_argument('mps', action='store_true', default=True, \
                    help='disables macOS GPU training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', \
                    help='how many batches to wait before logging training status')
parser.add_argument('--data_folder', type=str, default="./data", metavar='N', \
                    help='how many batches to wait before logging training status')
parser.add_argument('--seed', type=int, default=1, metavar='N', \
                    help='random number seed')
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)

os.makedirs(ARGS.path, exist_ok=True)
os.makedirs(ARGS.path_reconstructed_images, exist_ok=True)
os.makedirs(ARGS.path_generated_images, exist_ok=True)

# Sets the cuda and mps attributes of the args object based on
# the provided command-line arguments. If --no-cuda is not specified and
# CUDA is available, args.cuda is set to True.
# Similarly, if --no-mps is not specified and MPS (macOS GPU training) is
# available, args.mps is set to True.
ARGS.cuda = ARGS.cuda and torch.cuda.is_available()
ARGS.mps = ARGS.mps and torch.backends.mps.is_available()

# Defines a dictionary kwargs with additional keyword arguments
# for data loaders. If args.cuda is True, it includes
# 'num_workers': 1 and 'pin_memory': True. These settings are commonly used
# for efficient data loading when using CUDA.
ARGS.kwargs = {'num_workers': 1, 'pin_memory': True} if ARGS.cuda else {}

# Determines the PyTorch device (cuda, mps, or cpu)
# based on the provided command-line arguments.
if ARGS.cuda:
    ARGS.device = torch.device("cuda")
elif ARGS.mps:
    ARGS.device = torch.device("mps")
else:
    ARGS.device = torch.device("cpu")

ARGS.classes = "0123456789"

    