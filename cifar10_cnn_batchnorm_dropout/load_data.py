from torchvision import datasets, transforms
from torch.utils import data 

from global_name_space import ARGS


def load_data():
    # Define a series of transformations to be applied to the CIFAR-10 images.
    # transforms.Compose([...]): This function creates
    # a series of image transformations. In this case, it combines
    # two transformations: transforms.ToTensor() converts the images
    # to PyTorch tensors, and transforms.Normalize(...) normalizes
    # the pixel values of the tensors.
    # The Normalize transformation subtracts the mean (0.5 for each channel)
    # and divides by the standard deviation (0.5 for each channel).
    # This normalization ensures that the pixel values are in the range [-1, 1].
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
    ])

    # Load the training set with the specified transformations.
    # torchvision.datasets.CIFAR10(...): This function loads
    # the CIFAR-10 dataset. The transform argument is set to
    # the composed transformation, which means that
    # the specified transformations will be applied to each image in the dataset.
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create a data loader for the training set.
    # torch.utils.data.DataLoader(...): This function creates
    # a PyTorch data loader for the dataset. It allows efficient loading of
    # data in batches during training or testing. The shuffle parameter is set
    # to True for the training loader,
    # which shuffles the order of the data in each epoch.
    # The num_workers parameter specifies the number of subprocesses
    # used for data loading.
    trainloader = data.DataLoader(trainset, **ARGS.train_kwargs)
    testloader = data.DataLoader(testset, **ARGS.test_kwargs)

    return trainloader, testloader