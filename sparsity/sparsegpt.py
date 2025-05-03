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
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W_flat = W.flatten(1)
            self.rows, self.columns = W_flat.shape
        elif isinstance(layer, transformers.Conv1D):
            W_flat = W.t()
            self.rows, self.columns = W_flat.shape
        else:
            self.rows, self.columns = W.shape

        self.alpha = alpha
        self.beta = beta
        self.sum_a = torch.zeros(self.columns, device=self.dev)
        self.sum_outer = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out=None):
        # reshape input to [samples, in_features]
        x = inp
        if isinstance(self.layer, nn.Conv2d):
            N, C, H, W = x.shape
            x = x.permute(0,2,3,1).reshape(-1, C)
        elif isinstance(self.layer, transformers.Conv1D):
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
        elif isinstance(self.layer, nn.Linear):
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])
        else:
            raise NotImplementedError
        assert x.shape[1] == self.columns
        n = x.shape[0]
        self.sum_a += x.sum(dim=0)
        self.sum_outer += x.t().matmul(x)
        self.nsamples += n

    def fasterprune(self, sparsity_ratio, prunen=0, prunem=0,
                    blocksize=128, percdamp=0.01, disable_update=False):
        # build Frechet Hessian
        mu = self.sum_a / self.nsamples
        Sigma = (self.sum_outer / self.nsamples) - mu.unsqueeze(1).matmul(mu.unsqueeze(0))
        H = self.alpha * mu.unsqueeze(1).matmul(mu.unsqueeze(0)) + self.beta * Sigma

        # extract weight matrix
        W = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d): W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D): W = W.t()

        M = torch.ones_like(W, dtype=torch.bool, device=self.dev)
        Losses = torch.zeros(self.rows, device=self.dev)

        # damping & factorize
        damp = percdamp * torch.mean(torch.diagonal(H))
        idx = torch.arange(self.columns, device=self.dev)
        H[idx, idx] += damp
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(H_inv, upper=True)

        # block-wise OBS-like prune
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            Wb = W[:, i1:i2].clone()
            Hb = Hinv[i1:i2, i1:i2]

            # score & mask
            if prunen == 0:
                scores = (Wb**2) / (torch.diagonal(Hb)**2).reshape(1, -1)
                thresh = torch.quantile(scores, sparsity_ratio)
                mask_b = scores <= thresh
            else:
                mask_b = torch.zeros_like(Wb, dtype=torch.bool)

            Err = torch.zeros_like(Wb)
            Loss_b = torch.zeros_like(Wb)
            for j in range(Wb.shape[1]):
                wj = Wb[:,j]
                mj = mask_b[:,j]
                d = Hb[j,j]
                qj = wj.clone()
                qj[mj] = 0.0
                M[:, i1+j] = ~mj
                Loss_b[:,j] = (wj - qj)**2 / d**2
                if not disable_update:
                    ej = (wj - qj) / d
                    Wb[:, j:] -= ej.unsqueeze(1).matmul(Hb[j, j:].unsqueeze(0))
                    Err[:,j] = ej
            if not disable_update:
                W[:, i1:i2] = Wb
            Losses += Loss_b.sum(dim=1) / 2
            if not disable_update:
                W[:, i2:] -= Err.matmul(Hinv[i1:i2, i2:])

        # write back
        if isinstance(self.layer, transformers.Conv1D): W = W.t()
        if isinstance(self.layer, nn.Conv2d): W = W.reshape(self.layer.weight.shape)
        self.layer.mask.data = M.to(self.layer.weight.dtype)
        if not disable_update:
            self.layer.weight.data = W.to(self.layer.weight.dtype)

        print(f"Frechet-SparseGPT: total error {Losses.sum().item():.4f}")

