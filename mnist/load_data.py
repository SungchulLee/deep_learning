from torchvision import datasets, transforms
from torch.utils import data

from global_name_space import ARGS


def load_data():
    # Define a transformation to convert images to PyTorch tensors
    transform = transforms.ToTensor()

    # Load the training set with the specified transformations.
    # torchvision.datasets.MNIST(...): This function loads the MNIST dataset.
    # The root parameter specifies the directory where the data will be downloaded.
    # The train parameter is set to True for the training set.
    # The download parameter is set to True to download the dataset if not already present.
    # The transform argument is set to the composed transformation,
    # which means that the specified transformations will be applied to each image in the dataset.
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Load the test set with the same transformations as the training set.
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create a data loader for the training set.
    # torch.utils.data.DataLoader(...): This function creates a PyTorch data loader for the dataset.
    # It allows efficient loading of data in batches during training or testing.
    # The DataLoader is configured with the parameters specified in ARGS.train_kwargs.
    # This includes options like batch_size, shuffle, and num_workers.
    # For the training loader, shuffle is set to True to shuffle the order of the data in each epoch.
    trainloader = data.DataLoader(trainset, **ARGS.train_kwargs)

    # Create a data loader for the test set with parameters from ARGS.test_kwargs.
    testloader = data.DataLoader(testset, **ARGS.test_kwargs)

    return trainloader, testloader