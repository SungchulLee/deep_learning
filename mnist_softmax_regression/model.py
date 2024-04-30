import torch
import torch.nn as nn

from global_name_space import ARGS


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with input size ARGS.input_size and output size len(ARGS.classes)
        #                      input_size (784) output_size (10)
        #                           |                 |
        #                           |                 |
        #                           |                 |
        #                           V                 V
        self.layer = nn.Linear(ARGS.input_size, len(ARGS.classes))

    def forward(self, x): 
    # def forward( self, (64,1,28,28) )

        # Flatten each image along the second dimension
        x = torch.flatten(x, 1)  
        # (64,784) = torch.flatten( (64,1,28,28), 1)
        
        # Apply the linear layer to each flattened image
        x = self.layer(x) 
        # (64,10) = self.layer( (64,784) )
        
        return x