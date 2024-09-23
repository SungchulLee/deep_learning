import matplotlib.pyplot as plt

from load_data import load_data

def main():
    trainloader, _ = load_data()

    for photo, label in trainloader:
        fig, axs = plt.subplots(8,8)
        for ax, img, label in zip(axs.reshape((-1,)), photo, label):
            ax.imshow(img.squeeze(), cmap="binary")
            ax.axis("off")
            ax.set_title(f"Label : {label}")
        plt.tight_layout()
        plt.show()

        break


if __name__ == "__main__":
    main()