from torchvision import datasets, transforms
from torch.utils import data

from global_name_space import ARGS

def load_data():
    # Define a transformation to convert images to PyTorch tensors
    """
    The `transforms.Compose` function is used to create a sequence of image transformations. 
    Each transformation in the list will be applied sequentially to the input images. 
    Let's break down the transformations in detail:

    1. **ToTensor Transformation:**
        ```python
        transforms.ToTensor()
        ```
        This transformation converts the input images into PyTorch tensors. 
        It changes the image data from its original format (usually a NumPy array or PIL Image) into a PyTorch tensor. 
        PyTorch tensors are the primary data type used for neural network computations.

    2. **Normalize Transformation:**
        ```python
        transforms.Normalize([0.5], [0.5])
        ```
        This transformation normalizes the pixel values of the input images. 
        The `Normalize` transformation takes two parameters: the mean and standard deviation. 
        In this case, it subtracts 0.5 from each pixel value and then divides by 0.5. 
        This normalization is a common practice in deep learning to bring pixel values into a range 
        that is more suitable for training neural networks.
        If original pixel values are in [0,1], then after the normalization pixel values are in [-1,1].

    Putting it all together using `transforms.Compose`:
    ```python
    transform = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ] )
    ```
    This composition of transformations will convert it to a PyTorch tensor, and then normalize its pixel values. 
    It's a common preprocessing pipeline used when working with image data in deep learning applications.
    """
    # Because of transforms.Normalize([0.5], [0.5]),
    # the image value range is changed from [0,1] to [-1,1].
    #
    # img_after = transforms.Normalize( img_before )
    # img_after = ( img_before - mu ) / sigma
    #                                                                              mu     sigma of the original data distribution
    #                                                                               |      |
    #                                                                               |      |
    #                                                                               |      |
    #                                                                               V      V                                                                     
    transform = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ] )
    
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