import argparse
import os
import torch

parser = argparse.ArgumentParser(description='gan_mnist')
parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument("--b1", type=float, default=0.5, 
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, 
                    help="adam: decay of second order momentum of gradient")
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument("--latent_dim", type=int, default=100, 
                    help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, 
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, 
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, 
                    help="interval betwen image samples")
parser.add_argument('--cuda', type=bool, default=True,
                    help='ables CUDA training (default: False)')
parser.add_argument('--mps', action='store_true', default=True,
                    help='ables macOS GPU training (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)

ARGS.use_cuda = ARGS.cuda and torch.cuda.is_available()
ARGS.use_mps = ARGS.mps and torch.backends.mps.is_available()
if ARGS.use_cuda:
    ARGS.device = torch.device("cuda")
    ARGS.tensor = torch.cuda.FloatTensor
# elif ARGS.use_mps:
#     ARGS.device = torch.device("mps")
#     ARGS.tensor = torch.FloatTensor
else:
    ARGS.device = torch.device("cpu")
    ARGS.tensor = torch.FloatTensor

ARGS.train_kwargs = {'batch_size': ARGS.batch_size}
ARGS.test_kwargs = {'batch_size': ARGS.test_batch_size}
if ARGS.use_cuda:
    cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    ARGS.train_kwargs.update(cuda_kwargs)
    ARGS.test_kwargs.update(cuda_kwargs)

ARGS.classes = ("0","1","2","3","4","5","6","7","8","9")

os.makedirs("./model",exist_ok=True)
ARGS.path_g = './model/model_g.pth'
ARGS.path_d = './model/model_d.pth'

os.makedirs("./images", exist_ok=True)
os.makedirs("./samples", exist_ok=True)

ARGS.BCE_Loss = True # use nn.BCELoss
# ARGS.BCE_Loss = False # use F.binary_cross_entropy

"""
[Warning](https://pytorch.org/docs/stable/tensors.html)

torch.tensor() always copies data. 
If you have a Tensor data and just want to change its requires_grad flag, 
use requires_grad_() or detach() to avoid a copy. 
If you have a numpy array and want to avoid a copy, use torch.as_tensor().
"""