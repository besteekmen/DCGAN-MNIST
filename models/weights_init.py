import torch.nn as nn

# Create a helper function to initialize weights
def weights_init(m):
    """Initialize trainable parameters:
    transposed convolution kernels with Gaussian distribution,
    and affine parameters in batch normalization.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)