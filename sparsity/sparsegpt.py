# -*- coding: utf-8 -*-
"""
Updated pruning pipeline to use Frechet-style metric instead of plain MSE.
Keeps original function interface of prune_sparsegpt.
"""
import math
import time

import torch
import torch.nn as nn
import transformers

# ---- Frechet-style SparseGPT variant ----
class SparseGPTFrechet:
    def __init__(self, layer, alpha=1.0, beta=1.0):
        """
        layer: nn.Linear, nn.Conv2d, or transformers.Conv1D
        alpha: weight on mean-matching term
        beta: weight on covariance-matching term
        """
        self.layer = layer
        self.dev = layer.weight.device
        # Determine input dimensions
        if isinstance(layer, nn.Conv2d):
            W_flat = layer.weight.data.flatten(1)
            self.rows, self.columns = W_flat.shape
        elif isinstance(layer, transformers.Conv1D):
            W_flat = layer.weight.data.t()
            self.rows, self.columns = W_flat.shape
        else:
            self.rows, self.columns = layer.weight.data.shape

        # Frechet accumulators
        self.alpha = alpha
        self.beta = beta
        self.sum_a = torch.zeros(self.columns, device=self.dev)
        self.sum_outer = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out=None):
        """Accumulate input statistics."""
        x = inp
        # Flatten according to layer type
        if isinstance(self.layer, nn.Conv2d):
            N, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, C)
        elif isinstance(self.layer, transformers.Conv1D) and x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif isinstance(self.layer, nn.Linear) and x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        assert x.shape[1] == self.columns, f"Expected in_features={self.columns}, got {x.shape[1]}"
        n = x.shape[0]
        self.sum_a += x.sum(dim=0)
        self.sum_outer += x.t().matmul(x)
        self.nsamples += n

    def fasterprune(self, sparsity_ratio, prunen=0, prunem=0,
                    blocksize=128, percdamp=0.01, disable_update=False):
        """
        Perform one-shot N:M pruning update using Frechet-style objective.
        """
        # Extract and prepare weight matrix W
        W = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        # Initialize mask and loss buffer
        M = torch.ones_like(W, dtype=torch.bool, device=self.dev)
        Losses = torch.zeros(self.rows, device=self.dev)

        # Build Frechet Hessian H = alpha*mu*mu^T + beta*Sigma
        mu = self.sum_a / self.nsamples
        Sigma = (self.sum_outer / self.nsamples) - mu.unsqueeze(1).matmul(mu.unsqueeze(0))
        H = self.alpha * mu.unsqueeze(1).matmul(mu.unsqueeze(0)) + self.beta * Sigma

        # Damping & factorization
        damp = percdamp * torch.mean(torch.diagonal(H))
        idx = torch.arange(self.columns, device=self.dev)
        H[idx, idx] += damp
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(H_inv, upper=True)

        mask = None
        # Block-wise OBS-like prune
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            M1 = M[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # Initial mask per block
            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diagonal(Hinv1).reshape((1, -1))) ** 2
                    flat = tmp.flatten()
                    k = int(flat.numel() * sparsity_ratio)
                    thresh = flat.kthvalue(k).values if k>0 else flat.max()+1
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)

            # Per-column prune & update
            for j in range(count):
                w = W1[:, j]
                m = M1[:, j]
                d = Hinv1[j, j]

                # structured pruning step
                if prunen != 0 and prunem > 0 and j % prunem == 0:
                    end = min(j + prunem, count)
                    tmp = W1[:, j:end] ** 2 / (torch.diagonal(Hinv1)[j:end].reshape((1, -1))) ** 2
                    idxs = torch.topk(tmp, prunen, dim=1, largest=False)[1]
                    mask1.scatter_(1, idxs + j, True)

                # apply mask
                q = w.clone()
                q[mask1[:, j]] = 0.0
                m[mask1[:, j]] = 0

                Q1[:, j] = q
                M1[:, j] = m
                Losses1[:, j] = (w - q) ** 2 / d ** 2

                if not disable_update:
                    err1 = (w - q) / d
                    W1[:, j:] -= err1.unsqueeze(1).matmul(Hinv1[j, j:].unsqueeze(0))
                    Err1[:, j] = err1

            if not disable_update:
                W[:, i1:i2] = Q1
            M[:, i1:i2] = M1
            Losses += torch.sum(Losses1, 1) / 2

            if not disable_update:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print(f"Frechet-SparseGPT time {time.time()-tick:.2f}")
        print(f"error {Losses.sum().item():.4f}")

        # Write back mask & weights
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        if isinstance(self.layer, nn.Conv2d):
            W = W.reshape(self.layer.weight.shape)
        self.layer.mask.data = M.to(self.layer.weight.dtype)
        if not disable_update:
            self.layer.weight.data = W.to(self.layer.weight.dtype)

    def free(self):
        """Release accumulated statistics to free memory."""
        self.sum_a = None
        self.sum_outer = None
        self.nsamples = 0
        torch.cuda.empty_cache()

