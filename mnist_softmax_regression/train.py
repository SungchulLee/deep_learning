import torch

from global_name_space import ARGS

def train(model, trainloader, loss_ftn, optimizer, scheduler):
    model.train()

    # Loop over the dataset for a specified number of epochs
    for epoch in range(ARGS.epochs):
        running_loss = 0.0
        # Iterate over batches of training data
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(ARGS.device), labels.to(ARGS.device)
            # (64,1,28,28), (64,) = (64,1,28,28).to(ARGS.device), (64,).to(ARGS.device)

            # Forward pass
            outputs = model(inputs)
            # (64,10) = model( (64,1,28,28) )
            
            # Compute the loss
            loss = loss_ftn(outputs, labels)
            # () = loss_ftn( (64,10), (64,) )
            
            # Backward pass and optimization
            optimizer.zero_grad() # Zero the parameter gradients
            loss.backward() # Compute the gradients
            optimizer.step() # Update the parameters
            
            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # Print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        if ARGS.scheduler:
                scheduler.step()

def compute_accuracy(model, testloader):
    model.eval()

    correct = 0
    total = 0

    # Disable gradient computation for inference
    with torch.no_grad():
        for images, labels in testloader:

            images, labels = images.to(ARGS.device), labels.to(ARGS.device)
            # (1000,1,28,28), (1000,) = (1000,1,28,28).to(ARGS.device), (1000,).to(ARGS.device)

            outputs = model(images)
            # (1000,10) = model( (1000,1,28,28) )

            _, predicted = torch.max(outputs.data, 1)
            # _, (1000,)  = torch.max( (1000,10), 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            # () += ( (1000,) == (1000,) ).sum().item()

    # Print overall accuracy
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # Prepare to count predictions for each class
    correct_pred = {class_name: 0 for class_name in ARGS.classes}
    total_pred = {class_name: 0 for class_name in ARGS.classes}

    # Again, no gradients needed
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(ARGS.device), labels.to(ARGS.device)
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