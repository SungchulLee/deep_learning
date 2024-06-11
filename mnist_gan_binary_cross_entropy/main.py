import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from global_name_space import ARGS
from load_data import load_data
from model import Generator, Discriminator
from utils import show_batch_or_ten_generated_images, create_video
from train import train

def save_model(generator, discriminator):
    # Save the trained model's state dictionary to a file
    torch.save(generator.state_dict(), ARGS.path_g)
    torch.save(discriminator.state_dict(), ARGS.path_d)

def load_model():
    # Create an instance of the model and load the saved state dictionary
    generator = Generator().to(ARGS.device)
    discriminator = Discriminator().to(ARGS.device)
    generator.load_state_dict(torch.load(ARGS.path_g))
    discriminator.load_state_dict(torch.load(ARGS.path_d))
    return generator, discriminator

def main():
    trainloader, _ = load_data()

    generator = Generator().to(ARGS.device)
    discriminator = Discriminator().to(ARGS.device)
    if ARGS.BCE_Loss:
        adversarial_loss = nn.BCELoss()
    else:
        adversarial_loss = F.binary_cross_entropy
    optimizer_G = optim.Adam(generator.parameters(), lr=ARGS.lr, betas=(ARGS.b1, ARGS.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=ARGS.lr, betas=(ARGS.b1, ARGS.b2))

    # Display sample images and predictions before training
    show_batch_or_ten_generated_images(generator)

    # Train the model
    train(trainloader, optimizer_G, optimizer_D, adversarial_loss, generator, discriminator)

    # Save and load the trained model
    save_model(generator, discriminator)
    generator, discriminator = load_model()

    # Display sample images and predictions after training
    show_batch_or_ten_generated_images(generator)

    input_folder = "images"
    output_path = "./videos/output_video.mp4"
    create_video(input_folder, output_path)

if __name__ == "__main__":
    main()
