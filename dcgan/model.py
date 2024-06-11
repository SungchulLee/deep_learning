import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from global_name_space import ARGS
from load_data import load_data


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, nc):
        super().__init__()
        nz = int(ARGS.nz)
        ngf = int(ARGS.ngf)

        self.ngpu = int(ARGS.ngpu)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc):
        super().__init__()
        ndf = int(ARGS.ndf)

        self.ngpu = ngpu = int(ARGS.ngpu)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


def show_batch_or_ten_generated_images(model_g):
    """
    show batch generated images
    if batch is larger than 10, show ten images instead 
    """
    z = torch.normal(mean=0., std=1., size=(ARGS.batchSize, ARGS.nz, 1, 1)).to(ARGS.device)
    outputs = model_g(z)

    fig, axes = plt.subplots(1, min(ARGS.batchSize,10), figsize=(12, 3))
        
    for i, image in enumerate(outputs):
        # This line performs a simple "unnormalization" of the image.
        #image = image / 2 + 0.5  # unnormalize

        # Convert the PyTorch tensor img to a NumPy array.
        # The numpy() method is used to convert a PyTorch tensor to a NumPy array.
        image = image.cpu().detach().numpy()

        # Transpose the dimensions of the NumPy array
        # to the order expected by Matplotlib for displaying images.
        # Matplotlib expects the channels to be the last dimension,
        # so the transpose rearranges the dimensions from (C, H, W) to (H, W, C).
        image = np.transpose(image, (1, 2, 0))

        # Display the image with its true label and predicted label
        axes[i].imshow(image,cmap="binary")
        axes[i].axis("off")

        if i == 9:
            break

    plt.show()


def main():
    dataloader, nc, classes = load_data()

    generator = Generator(nc).to(ARGS.device)
    discriminator = Discriminator(nc).to(ARGS.device)

    show_batch_or_ten_generated_images(generator)

if __name__ == "__main__":
    main()