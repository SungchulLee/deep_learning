import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from global_name_space import ARGS

def show_batch_or_ten_generated_images(generator):
    """
    show batch generated images
    if batch is larger than 10, show ten images instead 
    """
    z = torch.normal(mean=0.,std=1.,size=(ARGS.batch_size,ARGS.latent_dim)).type(ARGS.tensor).to(ARGS.device)
    # (64,100) = torch.normal(mean=0.,std=1.,size=(ARGS.batch_size,ARGS.latent_dim)).type(ARGS.tensor).to(ARGS.device)
     
    outputs = generator(z)
    # (64,1,28,28) = generator( (64,100) ) 

    _, axes = plt.subplots(1, min(ARGS.batch_size,10), figsize=(12, 3))
        
    for i, image in enumerate(outputs):
    # for i, (1,28,28) in enumerate( (64,1,28,28) ):
         
        # This line performs a simple "unnormalization" of the image.
        image = image / 2.0 + 0.5  # unnormalize
        # (1,28,28) = (1,28,28) / 2.0 + 0.5

        # Convert the PyTorch tensor img to a NumPy array.
        # The numpy() method is used to convert a PyTorch tensor to a NumPy array.
        image = image.cpu().detach().numpy()
        # (1,28,28) = (1,28,28).cpu().detach().numpy()

        # Transpose the dimensions of the NumPy array
        # to the order expected by Matplotlib for displaying images.
        # Matplotlib expects the channels to be the last dimension,
        # so the transpose rearranges the dimensions from (C, H, W) to (H, W, C).
        image = np.transpose(image, (1, 2, 0))
        # (28,28,1) = np.transpose( (28,28,1), (1,2,0) )

        # Display the image with its true label and predicted label
        axes[i].imshow(image,cmap="binary")
        axes[i].axis("off")

        if i == 9:
            break

    plt.show()

def resize_image(image, target_size):
    # Resize the image to the target size
    return cv2.resize(image, target_size)

def create_video(input_folder, output_path, fps=24, target_size=(160, 160)):
    images = []
    
    # Read PNG files from the input folder
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".png"):
            filepath = os.path.join(input_folder, filename)
            image = imageio.imread(filepath)

            # Resize the image to be divisible by 16
            resized_image = resize_image(image, target_size)
            images.append(resized_image)

    # Write the resized images to a video file
    imageio.mimsave(output_path, images, fps=fps)