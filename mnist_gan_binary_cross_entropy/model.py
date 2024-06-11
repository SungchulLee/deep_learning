import numpy as np
import torch.nn as nn

from global_name_space import ARGS


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_shape = (ARGS.channels, ARGS.img_size, ARGS.img_size)
        # self.img_shape = (1,28,28) 

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(ARGS.latent_dim, 128, normalize=False), # (64,100) ---> (64,128)
            *block(128, 256), # (64,128) ---> (64,256)
            *block(256, 512), # (64,256) ---> (64,512)
            *block(512, 1024), # (64,512) ---> (64,1024)
            nn.Linear(1024, int(np.prod(self.img_shape))), # (64,1024) ---> (64,784)
            nn.Tanh() # (64,784) ---> (64,784)
            # Because of the nn.Tanh() unit
            # the image value range is [-1,1] , not [0,1]. 
        )

    def forward(self, z):
    # def forward(self, (64,100) ):
        img = self.model(z)
        # (64,784) = self.model( (64,100) ) 

        img = img.view(img.size(0), *self.img_shape)
        # (64,1,28,28) = (64,784).view(img.size(0), *self.img_shape)
         
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_shape = (ARGS.channels, ARGS.img_size, ARGS.img_size)
        # self.img_shape = (1,28,28) 

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512), # (64,784) ---> (64,512)
            nn.LeakyReLU(0.2, inplace=True), # (64,512) ---> (64,512)
            nn.Linear(512, 256), # (64,512) ---> (64,256)
            nn.LeakyReLU(0.2, inplace=True), # (64,256) ---> (64,256)
            nn.Linear(256, 1), # (64,256) ---> (64,1)
            nn.Sigmoid(), # (64,1) ---> (64,1)
            # Because of the nn.Sigmoid() unit
            # the final output range is [0,1],
            # which represents the probability of valid. 
        )

    def forward(self, img):
    # def forward(self, (64,1,28,28) ):
        img_flat = img.view(img.size(0), -1)
        # (64,784) = (64,1,28,28).view(img.size(0), -1) 

        validity = self.model(img_flat)
        # (64,1) = self.model( (64,784) ) 
        
        return validity
    
"""
The `nn.BatchNorm1d` layer in PyTorch is a batch normalization layer 
designed for one-dimensional input, often used in the context of neural networks. 
The `BatchNorm1d` layer normalizes the input along a specific dimension 
(usually the feature dimension) within each mini-batch during training. 
Let's break down the parameters used in the provided code: 
`nn.BatchNorm1d(128, 0.2)`.

- 128 : This parameter represents the number of features 
in the input tensor that will be normalized. 
In the context of the provided GAN code, 
this is the number of output features from the linear layer 
that precedes the batch normalization. 
If the input shape is (64, 128) and 
you are applying batch normalization along the second dimension 
(assuming that it's the feature dimension), 
you would compute the mean and standard deviation 
for each of the 128 features independently.
So, you are computing 128 means and standard deviations for the normalization step.

- `0.2`: This parameter is the momentum used for 
the running mean and standard deviation computation during training. 
It determines the weight given to the previous running statistics 
when updating them with the current mini-batch statistics. 
A value of 0.2 means that 80% of the running statistics 
come from the previous update, and 20% come from the current mini-batch.
Default is 0.1.
This momentum argument is different from one used 
in optimizer classes and the conventional notion of momentum. 
Mathematically, the update rule for running statistics here is 
$$
x_{new} = (1-momentum) * x_{moving_average} + momentum * x_{current}
$$
[src](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)

Batch normalization helps stabilize and speed up the training of neural networks. 
It works by normalizing the input to a layer, 
reducing internal covariate shift, 
and enabling the use of higher learning rates. 
Additionally, it introduces learnable parameters (gamma and beta) 
that allow the model to adapt the normalization to the specific needs of the data.

Here's a brief overview of how batch normalization works:

1. **Normalization:** It normalizes the input along the specified dimension 
(in this case, the feature dimension 128) 
by subtracting the mean and dividing by the standard deviation.
If the input shape is (64, 128) and 
you are applying batch normalization along the second dimension 
(assuming that it's the feature dimension), 
you would compute the mean and standard deviation 
for each of the 128 features independently.
So, you are computing 128 means and standard deviations for the normalization step.

2. **Scaling and Shifting:** The normalized values are then scaled and shifted 
using learnable parameters `gamma` (scaling) and `beta` (shifting). 
These parameters are learned during training.

In the provided code, a batch normalization layer is applied 
after a linear layer in the generator's architecture. 
The choice of applying batch normalization depends on the design choices 
made by the model architect and the specific requirements of the GAN training.
"""

"""
Q. For the train mode normalization, mini-batch mean and std are used? 
For the eval mode normalization, running mean and std are used?

A. Yes, that's correct.

- **Training Mode Normalization:**
  - During the training mode, the batch normalization layer normalizes 
  the input using the mean and standard deviation 
  computed from the current mini-batch. 
  These are often referred to as the mini-batch mean and standard deviation. 
  The purpose is to normalize the activations within the current mini-batch 
  and facilitate training.

- **Evaluation Mode Normalization:**
  - During the evaluation mode, the batch normalization layer uses 
  the accumulated running mean and standard deviation 
  that were computed during the training phase. 
  The running mean and standard deviation are fixed and used for normalization 
  during evaluation. 
  This ensures that the model's behavior remains consistent with 
  what it learned during training.

In summary:
- **Training:** Mini-batch mean and standard deviation.
- **Evaluation:** Running mean and standard deviation.
"""

"""
Q. Is inplace=True needed in nn.LeakyReLU(0.2, inplace=True)?

A. The `inplace=True` argument in `nn.LeakyReLU` is optional and controls 
whether the operation is performed in-place or returns a new tensor. 
Setting `inplace=True` means that the operation modifies the input tensor itself, 
while `inplace=False` (or omitting it) creates a new tensor 
and leaves the original tensor unchanged.

In the provided GAN code, `nn.LeakyReLU(0.2, inplace=True)` is used 
with `inplace=True`. Using `inplace=True` can be more memory-efficient, 
as it avoids creating a new tensor. 
However, it's important to note that modifying tensors in-place can have 
implications, especially if you plan to use the original tensor later in your code.

If memory efficiency is not a major concern and you want to avoid potential issues 
with in-place operations, you can omit `inplace=True` or set it to `False`. 
The default behavior is `inplace=False`. The modified line would be:

```python
nn.LeakyReLU(0.2)
```

In most cases, whether to use `inplace=True` depends on your specific use case 
and whether you need to retain the original tensor for further computations. 
If you're unsure, it's often safe to omit `inplace=True`.
"""

"""
Q. I am worrying about the gradient computation. Is it ok in this regard?

A. The use of `inplace=True` in `nn.LeakyReLU(0.2, inplace=True)` affects 
the gradient computation in the sense that it modifies the input tensor in-place. 
While this can be more memory-efficient, it has some implications:

1. **In-place Operations and Autograd:**
   - In-place operations can sometimes interfere 
   with PyTorch's automatic differentiation (autograd) system.
   - If the original tensor is modified in-place and used later 
   in the computation graph, it might lead to unexpected behavior, 
   as the autograd system relies on the original tensor's information 
   during backpropagation.

2. **Memory Efficiency vs. Gradient Computation:**
   - In-place operations can save memory by modifying the existing tensor, 
   but they come with potential pitfalls related to gradient computation.
   - If you encounter any issues with gradients or 
   if you plan to use the original tensor later in your computation, 
   it might be safer to avoid in-place operations.

To ensure more predictable behavior with autograd and to avoid potential issues 
with gradient computation, you can use `inplace=False` or simply omit it:

```python
nn.LeakyReLU(0.2)
```

This way, a new tensor will be created, and the original tensor will remain 
unchanged. It's a safer choice if you're concerned about autograd behavior 
and potential gradient-related issues.
"""

"""
Q. So, inplace=True is ok for the gradient computation 
if the node of nn.LeakyRelu is not called elsewhere.

A. Yes, that's correct. 
If the result of `nn.LeakyReLU(0.2, inplace=True)` is not used elsewhere 
in your code (i.e., if the node in the computation graph is not referenced later), 
and you don't need to retain the original tensor for subsequent computations, 
using `inplace=True` is generally okay in terms of gradient computation.

In this scenario, the in-place operation won't interfere 
with later gradient computation because the gradient information is not needed 
beyond the current node in the computation graph. 
Using `inplace=True` in such cases can be more memory-efficient.

Just be mindful of the specific context and ensure 
that the tensor modified in-place is not needed 
for subsequent operations or backpropagation. 
If you're uncertain or if you plan to use the original tensor later, 
you may choose to omit `inplace=True` or set it to `False` 
to avoid potential issues.
"""