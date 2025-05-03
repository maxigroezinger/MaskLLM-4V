import math
import time

import torch
import torch.nn as nn
import transformers


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPTFrechet:
    def __init__(self, layer, alpha=1.0, beta=1.0):
        """
        layer: nn.Linear, nn.Conv2d, or transformers.Conv1D
        alpha: weight on mean-matching term
        beta: weight on covariance-matching term
        """
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        # determine rows (output dim) and columns (input dim)
        if isinstance(layer, nn.Conv2d):
            # flatten weight to [out_features, in_features]
            W_flat = W.flatten(1)
            self.rows, self.columns = W_flat.shape
        elif isinstance(layer, transformers.Conv1D):
            # Conv1D stores weight as [in, out]
            W_flat = W.t()
            self.rows, self.columns = W_flat.shape
        else:
            # nn.Linear
            self.rows, self.columns = W.shape

        # Frechet accumulators
        self.alpha = alpha
        self.beta = beta
        self.sum_a = torch.zeros(self.columns, device=self.dev)
        self.sum_outer = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out=None):
        """
        Accumulate input statistics.  inp may be:
         - [batch, in_features] for Linear
         - [batch, seq_len, in_features] for Conv1D
         - [batch, in_channels, H, W] for Conv2d
        """
        # reshape input to [num_samples, in_features]
        x = inp
        if isinstance(self.layer, nn.Conv2d):
            # permute to [batch, H, W, C] then flatten
            N, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, C)
        elif isinstance(self.layer, transformers.Conv1D):
            # x: [batch, seq_len, in]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
        elif isinstance(self.layer, nn.Linear):
            if x.dim() == 2:
                # [batch, in]
                pass
            else:
                # maybe [batch, seq_len, in]
                x = x.reshape(-1, x.shape[-1])
        else:
            raise NotImplementedError(f"Unsupported layer type {type(self.layer)}")

        # x now [samples, in_features]
        assert x.shape[1] == self.columns, f"Expected in_features={self.columns}, got {x.shape[1]}"
        samples = x.shape[0]
        # accumulate
        self.sum_a += x.sum(dim=0)
        self.sum_outer += x.t().matmul(x)
        self.nsamples += samples

    def fasterprune(self, sparsity_ratio, prunen=0, prunem=0,
                    blocksize=128, percdamp=0.01, disable_update=False):
        """
        Perform one-shot N:M pruning update using Frechet-style objective.
        sparsity_ratio: fraction of weights to prune in each block
        prunen, prunem: optional structured pruning arguments
        blocksize: number of columns to process per chunk
        percdamp: damping fraction on diagonal of Frechet Hessian
        disable_update: if True, only compute masks without updating weights
        """
        # Build Frechet Hessian: H = alpha * mu mu^T + beta * Sigma
        mu = self.sum_a / self.nsamples  # [in_features]
        Sigma = self.sum_outer / self.nsamples - mu.unsqueeze(1).matmul(mu.unsqueeze(0))
        H = self.alpha * mu.unsqueeze(1).matmul(mu.unsqueeze(0)) + self.beta * Sigma

        # Extract weight matrix W_flat: [out_features, in_features]
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # Initialize mask M and losses tracking
        M = torch.ones_like(W, dtype=torch.bool, device=self.dev)
        Losses = torch.zeros(self.rows, device=self.dev)

        # Add damping to H for numerical stability
        damp = percdamp * torch.mean(torch.diagonal(H))
        diag_idx = torch.arange(self.columns, device=self.dev)
        H[diag_idx, diag_idx] += damp

        # Compute H_inv^{1/2} for block updates
        # cholesky H -> lower L, then invert, then cholesky for upper R
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
        R = torch.linalg.cholesky(H_inv, upper=True)
        Hinv = R

        # Block-wise one-shot solve (same pattern as SparseGPT)
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            W_block = W[:, i1:i2].clone()
            M_block = M[:, i1:i2].clone()
            Hinv_block = Hinv[i1:i2, i1:i2]
            block_count = i2 - i1
            
            # Determine mask per-column if not provided
            if prunen == 0:
                scores = (W_block**2) / (torch.diagonal(Hinv_block)**2).reshape(1, -1)
                thresh = torch.quantile(scores, sparsity_ratio)
                mask_block = scores <= thresh
            else:
                # legacy prunem-based mask
                mask_block = torch.zeros_like(W_block, dtype=torch.bool)

            # Apply mask and compute updates
            Err = torch.zeros_like(W_block)
            Loss_block = torch.zeros_like(W_block)
            for j in range(block_count):
                wj = W_block[:, j]
                mj = mask_block[:, j]
                pj = ~mj  # keep positions
                d = Hinv_block[j, j]

                # prune: set wj[mj] = 0
                qj = wj.clone()
                qj[mj] = 0.0
                M_block[:, j] = pj
                Loss_block[:, j] = (wj - qj)**2 / d**2

                if not disable_update:
                    errj = (wj - qj) / d
                    # rank-1 update to remaining columns
                    W_block[:, j:] -= errj.unsqueeze(1).matmul(Hinv_block[j, j:].unsqueeze(0))
                    Err[:, j] = errj

            # write back
            if not disable_update:
                W[:, i1:i2] = W_block
            M[:, i1:i2] = M_block
            Losses += Loss_block.sum(dim=1) / 2

            if not disable_update:
                # propagate error to future blocks
                W[:, i2:] -= Err.matmul(Hinv[i1:i2, i2:])

        # Copy mask & weights back into layer
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        if isinstance(self.layer, nn.Conv2d):
            W = W.reshape(self.layer.weight.shape)
        self.layer.mask.data = M.to(self.layer.weight.dtype)
        if not disable_update:
            self.layer.weight.data = W.to(self.layer.weight.dtype)

        # report
        print(f"Frechet-SparseGPT: total error {Losses.sum().item():.4f}")
