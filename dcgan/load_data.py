import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils import data

from global_name_space import ARGS


def load_data():
    if ARGS.dataset in ['imagenet', 'folder', 'lfw']:
        classes = None
        dataset = datasets.ImageFolder(root=ARGS.dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(ARGS.imageSize),
                                    transforms.CenterCrop(ARGS.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        nc=3
    elif ARGS.dataset == 'lsun':
        classes = [ c + '_train' for c in ARGS.classes.split(',')]
        dataset = datasets.LSUN(root=ARGS.dataroot, classes=classes,
                            transform=transforms.Compose([
                                transforms.Resize(ARGS.imageSize),
                                transforms.CenterCrop(ARGS.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc=3
    elif ARGS.dataset == 'cifar10':
        classes = None
        dataset = datasets.CIFAR10(root=ARGS.dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(ARGS.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc=3
    elif ARGS.dataset == 'mnist':
        classes = None
        dataset = datasets.MNIST(root=ARGS.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(ARGS.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1
    elif ARGS.dataset == 'fake':
        classes = None
        dataset = datasets.FakeData(image_size=(3, ARGS.imageSize, ARGS.imageSize),
                                transform=transforms.ToTensor())
        nc=3

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=ARGS.batchSize,
                                            shuffle=True, num_workers=int(ARGS.workers))
    return dataloader, nc, classes


def show_batch_or_ten_images_with_label(dataloader):
    """
    show batch images with label
    if batch is larger than 10, show ten images instead 
    """
    # Iterate through batches of images and labels in the dataloader
    for images, labels in dataloader:
        # Create a subplot with a horizontal arrangement of images
        fig, axes = plt.subplots(1, min(ARGS.batchSize,10), figsize=(12, 3))

        # Iterate through individual images and labels in the batch
        for i, (image, label) in enumerate(zip(images, labels)):
            # This line performs a simple "unnormalization" of the image.
            image = image / 2 + 0.5  # unnormalize

            # Convert the PyTorch tensor img to a NumPy array.
            # The numpy() method is used to convert a PyTorch tensor to a NumPy array.
            image = image.permute(1,2,0).numpy().squeeze() # (3, 64, 64) ---> (64, 64, 3)

            # Transpose the dimensions of the NumPy array
            # to the order expected by Matplotlib for displaying images.
            # Matplotlib expects the channels to be the last dimension,
            # so the transpose rearranges the dimensions from (C, H, W) to (H, W, C).
            # image = np.transpose(image, (1, 2, 0))

            # Display the image with its corresponding label
            axes[i].imshow(image, cmap="binary")
            axes[i].axis("off")

            if i == 9:
                break

        # Show the current batch of images
        plt.show()
        break  # Break after showing the first batch (remove for displaying all batches)


def main():
    dataloader, nc, classes = load_data()
    show_batch_or_ten_images_with_label(dataloader)


if __name__ == "__main__":
    main()


