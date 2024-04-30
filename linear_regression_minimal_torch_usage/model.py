import torch

def initialize_weight():
    w = torch.randn((1,), requires_grad=True) # (1,)
    b = torch.randn((1,), requires_grad=True) # (1,)
    return w, b

def predict(inputs, w, b):
    return inputs * w + b # (100, 1) * (1,) + (1,) = (100, 1)