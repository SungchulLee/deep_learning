import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from global_name_space import ARGS
from load_data import load_data
from model import initialize_weight, model


def train(trainloader, w, b):
    w_trace = []
    b_trace = []
    loss_trace = []
    for _ in range(ARGS.epochs):
        for inputs, targets in trainloader: 
            preds = model(inputs, w, b) # (100, 1)
            loss = mse(preds, targets) # ()
            w_trace.append(w.item())
            b_trace.append(b.item())
            loss_trace.append(loss.item())

            loss.backward()
            with torch.no_grad():
                w -= w.grad * ARGS.lr
                b -= b.grad * ARGS.lr
                w.grad.zero_()
                b.grad.zero_()

    return w, b, w_trace, b_trace, loss_trace


def mse(preds, targets):
    return torch.sum((preds-targets)**2) / targets.numel()

def save(w, b):
    with open(ARGS.path_w, 'wb') as f:
        pickle.dump(w, f)
    with open(ARGS.path_b, 'wb') as f:
        pickle.dump(b, f)


def load():
    with open(ARGS.path_w, 'rb') as f:
        w = pickle.load(f)
    with open(ARGS.path_b, 'rb') as f:
        b = pickle.load(f)
    return w, b


def main():
    trainloader, testloader = load_data()

    w, b = initialize_weight()

    w, b, w_trace, b_trace, loss_trace = train(trainloader, w, b)
    save(w, b)

    # visuaization of training process
    w, b = load()
    _, (ax0, ax1, ax2) = plt.subplots(1,3,figsize=(12,3))
    ax0.plot(loss_trace,label="loss")
    ax1.plot(w_trace,label="pred slope")
    ax1.plot(np.ones_like(w_trace)*2,'--r',label="true slope")
    ax2.plot(b_trace,label="pred intercept")
    ax2.plot(np.ones_like(b_trace),'--r',label="true intercept")
    for ax in (ax0, ax1, ax2):
        ax.legend()
    plt.tight_layout()
    plt.show()

    # visualization of final model performance
    inputs, targets = next(iter(testloader))
    preds = model(inputs, w, b)
    _, ax = plt.subplots(figsize=(12,3))
    ax.plot(inputs,targets,'k.',label="data")
    ax.plot(inputs,preds.detach().numpy(),'r-',label="pred")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()