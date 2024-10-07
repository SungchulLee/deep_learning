import torch
import torch.nn as nn
import torch.optim as optim

from global_name_space import ARGS
from load_data import load_data
from model import Net
from utils import show_batch_or_ten_images_with_label_and_predict

def train(model, trainloader, optimizer, loss_ftn):
    model.train()

    # Loop over the dataset for a specified number of epochs
    for epoch in range(ARGS.epochs):
        running_loss = 0.0
        # Iterate over batches of training data
        for i, (inputs, labels) in enumerate(trainloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = loss_ftn(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # Print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

def save_model(model):
    # Save the trained model's state dictionary to a file
    torch.save(model.state_dict(), ARGS.path)

def load_model():
    # Create an instance of the model and load the saved state dictionary
    model = Net()
    model.load_state_dict(torch.load(ARGS.path))
    return model

def compute_accuracy(model, testloader):
    model.eval()

    correct = 0
    total = 0

    # Disable gradient computation for inference
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print overall accuracy
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # Prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in ARGS.classes}
    total_pred = {classname: 0 for classname in ARGS.classes}

    # Again, no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            # Collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[ARGS.classes[label]] += 1
                total_pred[ARGS.classes[label]] += 1

    # Print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def main():
    trainloader, testloader = load_data()

    model = Net()
    loss_ftn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=ARGS.lr, momentum=ARGS.momentum)

    # Display sample images and predictions before training
    show_batch_or_ten_images_with_label_and_predict(testloader, model)

    # Train the model
    train(model, trainloader, optimizer, loss_ftn)

    # Display sample images and predictions after training
    show_batch_or_ten_images_with_label_and_predict(testloader, model)

    # Save and load the trained model
    save_model(model)
    model = load_model()

    # Compute and display accuracy on test data
    compute_accuracy(model, testloader)

if __name__ == "__main__":
    main()
