import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from global_name_space import ARGS
from load_data import load_data
from model import initialize_weight, predict

def compute_loss(preds, targets):
    return torch.sum((preds-targets)**2) / targets.numel()

def train(inputs, targets, w, b):
    w_trace = []
    b_trace = []
    loss_trace = []
    for _ in range(ARGS.epochs):
        preds = predict(inputs, w, b) # (100, 1)
        loss = compute_loss(preds, targets) # ()

        w_trace.append(w.item())
        b_trace.append(b.item())
        loss_trace.append(loss.item())

        loss.backward() # compute the gradients

        # Gradient Descent Update
        # torch.no_grad() is a context manager
        # that is used to turn off gradient computation temporarily.
        # Inside this context, PyTorch operations
        # won't build the computational graph for gradient computation.
        # This is useful in scenarios
        # where you have parameters (like weights)
        # that you don't want to update during the optimization process,
        # or when you are performing inference
        # and don't need to compute gradients.
        with torch.no_grad():
            w -= w.grad * ARGS.lr # in-place operation
            b -= b.grad * ARGS.lr # in-place operation

            # Zeroing Gradients:
            # After the weights and bias have been updated,
            # it's important to zero out the gradients
            # to prevent them from accumulating across multiple iterations.
            w.grad.zero_()
            b.grad.zero_()

    return w, b, w_trace, b_trace, loss_trace


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


def draw(loss_trace, w_trace, b_trace, inputs, targets, preds):
    _, (ax0, ax1, ax2, ax3) = plt.subplots(1,4,figsize=(15,4))
    ax0.plot(loss_trace,label="loss")
    ax1.plot(w_trace,label="pred slope")
    ax1.plot(np.ones_like(w_trace)*2,'--r',label="true slope")
    ax2.plot(b_trace,label="pred intercept")
    ax2.plot(np.ones_like(b_trace),'--r',label="true intercept")
    ax3.plot(inputs,targets,'k.',label="data")
    ax3.plot(inputs,preds.detach().numpy(),'r-',label="pred")
    for ax in (ax0, ax1, ax2, ax3):
        ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    trainloader = load_data() 
    inputs, targets = next(iter(trainloader))

    w, b = initialize_weight()
    w, b, w_trace, b_trace, loss_trace = train(inputs, targets, w, b)
    save(w, b)

    w, b = load()
    preds = predict(inputs, w, b)
    draw(loss_trace, w_trace, b_trace, inputs, targets, preds)


if __name__ == "__main__":
    main()