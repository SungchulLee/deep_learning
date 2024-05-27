import matplotlib.pyplot as plt
import torch

from global_name_space import ARGS

def show_ten_generated_images(model):
    z = torch.randn((10,20)).to(ARGS.device)
    images = model.decode(z).view(-1,28,28).to("cpu")

    fig, axes = plt.subplots(1, 10, figsize=(12, 3))
        
    for i, image in enumerate(images):
        # This line performs a simple "unnormalization" of the image.
        image = image / 2 + 0.5  # unnormalize

        # Convert the PyTorch tensor img to a NumPy array.
        # The numpy() method is used to convert a PyTorch tensor to a NumPy array.
        image = image.detach().numpy()

        # Display the image with its true label and predicted label
        axes[i].imshow(image,cmap="binary")
        axes[i].axis("off")

    plt.show()