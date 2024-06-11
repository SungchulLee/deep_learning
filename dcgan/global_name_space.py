import random
import os
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="mnist", help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--dataroot', default="./data", help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--real_samples_folder', default='./real_images', help='folder to output images and model checkpoints')
parser.add_argument('--fake_samples_folder', default='./fake_images', help='folder to output images and model checkpoints')
parser.add_argument('--modelf', default='./model', help='folder to output model checkpoints')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')
parser.add_argument('--colab', default=True, help='run colab')
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
ARGS = parser.parse_args()

if ARGS.dry_run:
    ARGS.niter = 1

# make folder to output images and model checkpoints
os.makedirs(ARGS.real_samples_folder, exist_ok=True)
os.makedirs(ARGS.fake_samples_folder, exist_ok=True)
os.makedirs(ARGS.modelf, exist_ok=True)

# set random number seed
if ARGS.manualSeed is None:
    ARGS.manualSeed = random.randint(1, 10000)
random.seed(ARGS.manualSeed)
torch.manual_seed(ARGS.manualSeed)

# set_device
ARGS.use_mps = ARGS.mps and torch.backends.mps.is_available()
if ARGS.cuda:
    ARGS.device = torch.device("cuda:0")
elif ARGS.use_mps:
    ARGS.device = torch.device("mps")
else:
    ARGS.device = torch.device("cpu")

# check_gpu_env
if torch.cuda.is_available() and not ARGS.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
elif torch.backends.mps.is_available() and not ARGS.mps:
    print("WARNING: You have mps device, to enable macOS GPU run with --mps")
else:
    print("gpu : ready to go")

# check_dataroot_env
if ARGS.dataroot is None and str(ARGS.dataset).lower() != 'fake':
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % ARGS.dataset)
else:
    print("dataroot : ready to go")

