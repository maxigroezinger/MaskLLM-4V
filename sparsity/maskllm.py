import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

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
        self._mask_options = generate_N_M_masks(N, M) 
        self.gate = nn.Parameter(torch.empty(
                self.weight.numel()//M, self._mask_options.size(0), device=self.weight.device, dtype=self.weight.dtype), requires_grad=True)
        torch.nn.init.normal_(self.gate, mean=0, std=gate_init_std)
        self.tau = 1
        self.scaling = scaling
        self.hard = hard
        self.register_buffer('mask', torch.ones((out_features, in_features), dtype=torch.float32))
        self.mask_oudated = False
        self.N = N
        self.M = M 

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, bias={self.bias is not None}, N={self.N}, M={self.M}, tau={self.tau}, scaling={self.scaling}, hard={self.hard})"

    def sparse_weight_reg(self):
        return self._sparse_weight_reg

    def forward(self, x):
        if self.training:
            self.mask_oudated = True # reset selected since we will update it
            soft_index = F.gumbel_softmax(self.gate * self.scaling, tau=self.tau, hard=self.hard, dim=1) # (Blocks x Candidate Masks)
            soft_mask = soft_index @ self._mask_options.to(x.device) # (Blocks x Candidate Masks) @ (Candidate Masks x M) = (Blocks x M)
            soft_mask = soft_mask.view(self.out_features, self.in_features)
            self._sparse_weight_reg = (self.weight.detach() * soft_mask).pow(2).sum()
            return F.linear(x, soft_mask * self.weight, self.bias)
        else:
            if self.mask_oudated: # for inference, we only compute the winner masks once for efficiency
                self._mask_options = self._mask_options.to(x.device)
                self.mask = self._mask_options[torch.argmax(self.gate, dim=1)].view(self.out_features, self.in_features)
                self.mask_oudated = False
            return F.linear(x, self.mask * self.weight, self.bias)

    def load_mask_prior(self, prior_strength=3):
        for name, param in self.named_parameters():
            with torch.no_grad():
                sparsity = (self.mask==0).sum().item() / self.mask.numel()
                # rank 0
                if torch.distributed.get_rank() == 0:
                    print(f"initializing {name} with prior (strength={prior_strength}), Prior Sparsity: {sparsity}")
                # prior will be the inner product the different candidates to the prior mask
                priors = (self._mask_options.unsqueeze(0) * self.mask.view(-1, 1, 4)).sum(dim=2) # (1, Candidate Masks, M) * (Blocks, 1, M) => Blocks x Candidate Masks
                self.gate.data += (priors-self.N//2) * self.gate.std() * prior_strength