import torch
import torch.nn as nn
import torch.optim as optim

from global_name_space import ARGS
from load_data import load_data
from model import Net
from train import train, compute_accuracy
from utils import show_batch_or_ten_images_with_label_and_predict

def save_model(model):
    # Save the trained model's state dictionary to a file
    torch.save(model.state_dict(), ARGS.path)

def load_model():
    # Create an instance of the model and load the saved state dictionary
    model = Net().to(ARGS.device)
    model.load_state_dict(torch.load(ARGS.path))
    return model

def main():
    trainloader, testloader = load_data()

    model = Net().to(ARGS.device)
    loss_ftn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=ARGS.lr, momentum=ARGS.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Display sample images and predictions before training
    show_batch_or_ten_images_with_label_and_predict(testloader, model)

    # Train the model
    train(model, trainloader, loss_ftn, optimizer, scheduler)

    # Display sample images and predictions after training
    show_batch_or_ten_images_with_label_and_predict(testloader, model)

    # Save and load the trained model
    save_model(model)
    model = load_model()

    # Compute and display accuracy on test data
    compute_accuracy(model, testloader)

if __name__ == "__main__":
    main()