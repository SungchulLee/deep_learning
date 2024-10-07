
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( # (16, 3 * 32 * 32)
            nn.Linear(32*32*3, 512), # (16, 512)
            nn.ReLU(), # (16, 512)
            nn.Linear(512, 512), # (16, 512)
            nn.ReLU(), # (16, 512)
            nn.Linear(512, 10), # (16, 10)
        )

    def forward(self, x): # (16, 3, 32, 32)
        x = self.flatten(x) # (16, 3 * 32 * 32)
        logits = self.linear_relu_stack(x)
        return logits

""" 
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        layers = []
        layers.append(nn.Linear(32*32*3, 20))
        layers.append(nn.ReLU())
        
        # Loop to add multiple linear layers and ReLU activations
        for _ in range(100):
            layers.append(nn.Linear(20, 20))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(20, 10))  # Final output layer
        
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
"""
