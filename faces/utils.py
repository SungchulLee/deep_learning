import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io

def show_one_image_from_datafolder(df_landmarks, n=65):
    img_name = df_landmarks.iloc[n, 0]
    image = io.imread(os.path.join('data/faces/', img_name))

    landmarks = df_landmarks.iloc[n, 1:]
    landmarks = np.asarray(landmarks)
    landmarks = landmarks.astype('float').reshape(-1, 2)

    _, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    ax.axis("off")
    plt.show()

def show_four_images_from_dataloader(dataloader):
    _, axes = plt.subplots(1,4,figsize=(12,3))
    i = -1
    for d in dataloader:
        images = d['image']
        landmarks = d['landmarks']
        for image, landmark in zip(images, landmarks):
            i += 1
            axes[i].imshow(image.permute(1,2,0))
            axes[i].scatter(landmark[:, 0], landmark[:, 1], s=10, marker='.', c='r')
            axes[i].axis("off")
            if i == 3:
                break
        if i == 3:
            break
    plt.tight_layout()
    plt.show()