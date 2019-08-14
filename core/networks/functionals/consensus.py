import torch


class AverageConsensus(torch.autograd.Function):
    """
    """
    def __init__(self, dim=1):
        """
        """
        self.dim = dim
        self.input_shape = None

    def forward(self, x): 
        output = x.mean(dims=self.dim)
        return output
    
    def backward(self, grad_out):
        grad_in = grad_out.unsqueeze(self.dim).expand(self.input_shape)
        grad_in = grad_in / float(self.input_shape[self.dim])
        return grad_in


class IdentityConsensus(torch.autograd.Function):
    """
    """
    def __init__(self, dim=1):
        """
        """
        self.dim = dim
        self.input_shape = None

    def forward(self, x):
        return x
    
    def backward(self, grad_out):
        return grad_out

