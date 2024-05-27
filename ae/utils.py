import matplotlib.pyplot as plt
import torch

from global_name_space import ARGS

def show_ten_generated_images(model):
    z = torch.randn((10,20)).to(ARGS.device)
    images = model.decode(z).view(-1,28,28).to("cpu")

    _, axes = plt.subplots(1, 10, figsize=(12, 3))
    for i, image in enumerate(images):
        image = image / 2 + 0.5  # unnormalize
        axes[i].imshow(image.detach().numpy(),cmap="binary")
        axes[i].axis("off")
    plt.show()