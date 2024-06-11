import cv2
import imageio.v2 as imageio
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from global_name_space import ARGS
from load_data import load_data
from model import Generator, Discriminator, weights_init


def set_and_initialize_model(nc):
    netG = Generator(nc).to(ARGS.device)
    netD = Discriminator(nc).to(ARGS.device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    return netG, netD


def save_model(netG, netD):
    torch.save(netG.state_dict(), ARGS.netG)
    torch.save(netD.state_dict(), ARGS.netD)


def load_model(netG, netD):
    netG.load_state_dict(torch.load(ARGS.netG))
    netD.load_state_dict(torch.load(ARGS.netD))


def train(netG, netD, dataloader, optimizerG, optimizerD, criterion, real_label, fake_label, start_epoch):
    nz = ARGS.nz
    fixed_noise = torch.randn(ARGS.batchSize, nz, 1, 1, device=ARGS.device)

    for epoch in range(start_epoch, ARGS.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(ARGS.device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                            dtype=real_cpu.dtype, device=ARGS.device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=ARGS.device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, ARGS.niter, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        f'{ARGS.real_samples_folder}/real_samples_epoch_{epoch:03d}.png',
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        f'{ARGS.fake_samples_folder}/fake_samples_epoch_{epoch:03d}.png',
                        normalize=True)

            if ARGS.dry_run:
                break
        # do checkpointing
        ARGS.netG = f'{ARGS.modelf}/netG_epoch_{epoch}.pth'
        ARGS.netD = f'{ARGS.modelf}/netD_epoch_{epoch}.pth'
        save_model(netG, netD)


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


def main():
    cudnn.benchmark = True
    real_label = 1
    fake_label = 0

    dataloader, nc, classes = load_data()

    netG, netD = set_and_initialize_model(nc)

    start_epoch = 0
    if ARGS.netG != '':
        load_model(netG, netD)
        start_epoch = int(ARGS.netG.split("_")[2].split(".")[0]) + 1
        print(start_epoch)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=ARGS.lr, betas=(ARGS.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=ARGS.lr, betas=(ARGS.beta1, 0.999))

    train(netG, netD, dataloader, optimizerG, optimizerD, criterion, real_label, fake_label, start_epoch)

    input_folder = "./fake_images"
    output_path = "./videos/output_video.mp4"
    create_video(input_folder, output_path)


if __name__ == "__main__":
    main()
