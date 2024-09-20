import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_N_M_masks(N, M):
    # Create all possible binary combinations N:M sparse masks
    combinations = list(itertools.combinations(range(M), N))
    # Create a tensor to store the result
    result = torch.zeros((len(combinations), M), dtype=torch.float32)
    # Fill in the ones according to the combinations
    for i, indices in enumerate(combinations):
        result[i, torch.tensor(indices)] = 1
    return result

class GumbelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, N=2, M=4, gate_init_std=0.02, tau=1, hard=False, scaling=1):
        super(GumbelLinear, self).__init__(in_features, out_features, bias)
        self._mask_options = generate_N_M_masks(in_features, out_features) 

        self.gate = Parameter(torch.empty(
                self.weight.numel()//M, self._mask_options.size(0), device=self.weight.device, dtype=self.weight.dtype), requires_grad=True)
        torch.nn.init.normal_(self.gate, mean=0, std=gate_init_std)
        self.tau = 1
        self.scaling = scaling

        self.mask = None # store the selected mask for inference

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, bias={self.bias is not None})"

    def forward(self, x):
        if self.training:
            self._mask_for_inference = None # reset selected since we will update it
            soft_index = F.gumbel_softmax(self.gate * self.scaling, tau=self.tau, hard=self.hard, dim=1) # (Blocks x Candidate Masks)
            soft_mask = soft_index @ self._mask_options # (Blocks x Candidate Masks) @ (Candidate Masks x M) = (Blocks x M)
            soft_mask = softmax_mask.view(self.out_features, self.in_features)
            return F.linear(x, soft_mask * self.weight, self.bias)
        else:
            if self.mask is None: # for inference, we only compute the winner masks once for efficiency
                self.mask = self._mask_options[torch.argmax(self.gate, dim=1)].view(self.out_features, self.in_features)
            return F.linear(x, self.mask * self.weight, self.bias)