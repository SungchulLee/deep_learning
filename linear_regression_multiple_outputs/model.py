import torch

def initialize_weight():
    w = torch.randn((2, 3), requires_grad=True) # (2, 3)
    b = torch.randn((2,), requires_grad=True) # (2,)
    return w, b

def predict(x, w, b):
    return x @ w.t() + b # (5, 3) @ (3, 2) + (2,) = (5, 2)