import matplotlib.pyplot as plt
import numpy as np
import torch

from global_name_space import ARGS

def show_batch_or_ten_images_with_label_and_predict(dataloader, model):
    """
    show batch images with label and predict
    if batch is larger than 10, show ten images instead 
    """
    for images, labels in dataloader:
    # for (64,1,28,28), (64,) in dataloader:

        # Make predictions using the provided model
        outputs = model(images.to(ARGS.device))
        # (64,10) = model( (64,1,28,28).to(ARGS.device) )

        # Get the predicted labels
        _, predicted_labels = torch.max(outputs, 1)
        # _, (64,) = torch.max( (64,10), 1 )

        _, axes = plt.subplots(1, min(ARGS.batch_size,10), figsize=(12, 3))
        
        for i, (image, label, predicted_label) in enumerate(zip(images, labels, predicted_labels.to("cpu"))):
        # for i, ( (1,28,28), (), () ) in enumerate(zip(images, labels, predicted_labels.to("cpu"))): 

            # Convert the PyTorch tensor img to a NumPy array.
            # The numpy() method is used to convert a PyTorch tensor to a NumPy array.
            image = image.numpy()
            # (1,28,28) = (1,28,28).numpy()

            # Transpose the dimensions of the NumPy array
            # to the order expected by Matplotlib for displaying images.
            # Matplotlib expects the channels to be the last dimension,
            # so the transpose rearranges the dimensions from (C, H, W) to (H, W, C).
            image = np.transpose(image, (1, 2, 0))
            # (28,28,1) = np.transpose( (1,28,28), (1,2,0) )

            # Display the image with its true label and predicted label
            axes[i].imshow(image,cmap="binary")
            axes[i].axis("off")
            axes[i].set_title("label: " + ARGS.classes[label] + "\npred: " + ARGS.classes[predicted_label])

            if i == 9:
                break

        plt.show()
        break