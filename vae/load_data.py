import torch
from torchvision import datasets, transforms

from global_name_space import ARGS

def load_data():
    # Creates a data loader for the MNIST training set
    # using torch.utils.data.DataLoader.
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(ARGS.data_folder, train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=ARGS.batch_size, shuffle=True, **ARGS.kwargs)

    # Creates a data loader for the MNIST test set in a similar manner.
    # It sets train=False to indicate that this is the test set.
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(ARGS.data_folder, train=False, transform=transforms.ToTensor()),
        batch_size=ARGS.batch_size, shuffle=False, **ARGS.kwargs)
    return train_loader, test_loader