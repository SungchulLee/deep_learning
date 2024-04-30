import matplotlib.pyplot as plt
import torch

from global_name_space import ARGS
from load_data import load_data
from model import initialize_weight, predict

def compute_loss(preds, targets):
    return torch.sum((preds-targets)**2) / targets.numel()

def train(trainloader, w, b):
    w_trace = []
    b_trace = []
    loss_trace = []
    for i in range(ARGS.epochs):
        for inputs, targets in trainloader:
          w_trace.append(w.detach().numpy())
          b_trace.append(b.detach().numpy())

          preds = predict(inputs, w, b) # (5, 2)
          loss = compute_loss(preds, targets) # ()
          loss_trace.append(loss.item())

          loss.backward()
          with torch.no_grad():
              w -= w.grad * ARGS.lr
              b -= b.grad * ARGS.lr
              w.grad.zero_()
              b.grad.zero_()

    return w, b, w_trace, b_trace, loss_trace


def draw(loss_trace):
    _, ax = plt.subplots(figsize=(15,4))
    ax.plot(loss_trace,label="loss")
    ax.legend()
    plt.show()


def main():
    trainloader = load_data()
    w, b = initialize_weight()
    w, b, *_, loss_trace = train(trainloader, w, b)
    draw(loss_trace)


if __name__ == "__main__":
    main()