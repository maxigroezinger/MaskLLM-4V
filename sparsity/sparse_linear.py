import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, bias={self.bias is not None})"

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)