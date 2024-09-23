import matplotlib.pyplot as plt
from load_data import load_data

def main():
    trainloader, _ = load_data()

    for imgs, labels in trainloader:
        _, axes = plt.subplots(8,8)
        for i in range(8):
            for j in range(8):
                axes[i,j].imshow(imgs[i*8+j].reshape((28,28)), cmap="binary")
                axes[i,j].spines["top"].set_visible(False)
                axes[i,j].spines["bottom"].set_visible(False)
                axes[i,j].spines["right"].set_visible(False)
                axes[i,j].spines["left"].set_visible(False)
                axes[i, j].set_xticks([])  # Remove x-ticks
                axes[i, j].set_yticks([])  # Remove y-ticks
                axes[i, j].set_title(f"Label : {labels[i*8+j]}")  # Remove y-ticks
        plt.tight_layout()
        plt.show()

        break

if __name__ == "__main__":
    main()