import math
import time

import torch
import torch.nn as nn
import transformers


DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:
    def __init__(self, layer, whiten=False, eps=1e-6):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        self.rows, self.columns = W.shape

        # second moment accumulator
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        # covariance accumulator for whitening
        self.C = torch.zeros_like(self.H)
        self.nsamples = 0

        self.whiten = whiten
        self.eps = eps

    def add_batch(self, inp, out):
        # prepare input matrix X: shape (batch, features)
        x = inp
        if x.ndim == 2:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.t()
        # scale
        scaled = math.sqrt(2 / (self.nsamples + batch)) * x.float()

        # update second moment H
        self.H *= self.nsamples / (self.nsamples + batch)
        self.H += scaled.matmul(scaled.t())

        # update covariance C for whitening
        self.C *= self.nsamples / (self.nsamples + batch)
        self.C += (x.float()).matmul(x.float().t())

        self.nsamples += batch

    def fasterprune(self, sparsity_ratio, prunen=0, prunem=0,
                    blocksize=128, percdamp=0.01, disable_update=False):
        # clone weight
        W = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        # compute whitening transforms if enabled
        if self.whiten:
            # regularize C and compute inverse-sqrt and sqrt
            C_reg = self.C / self.nsamples + self.eps * torch.eye(self.columns, device=self.dev)
            Lc = torch.linalg.cholesky(C_reg)
            C_inv_sqrt = torch.cholesky_inverse(Lc)
            C_sqrt = Lc
            # transform H and W to whitened feature space
            H = C_inv_sqrt.matmul(self.H).matmul(C_inv_sqrt)
            W = W.matmul(C_sqrt)
        else:
            H = self.H

        # pruning setup
        dead = torch.diag(H) == 0
        H = H.clone()
        H[dead, dead] = 1
        W[:, dead] = 0

        # damping
        damp = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(self.columns, device=self.dev)
        H[diag_idx, diag_idx] += damp

        # invert via Cholesky
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)

        M = torch.ones_like(W)
        Losses = torch.zeros(self.rows, device=self.dev)

        # blockwise pruning (unchanged)
        mask = None
        (0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone(); M1 = M[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                tmp = W1**2 / (torch.diag(Hinv1).reshape(1, -1))**2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel()*sparsity_ratio)]
                mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)

            for i in range(count):
                w = W1[:, i]; d = Hinv1[i, i]
                q = w.clone(); q[mask1[:, i]] = 0
                Q1[:, i] = q
                Losses1[:, i] = (w - q)**2 / d**2
                if not disable_update:
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
            if not disable_update:
                W[:, i1:i2] = Q1
            M[:, i1:i2] = M1
            Losses += torch.sum(Losses1, dim=1) / 2
            if not disable_update:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - time.time()))
        print('error', torch.sum(Losses).item())

        # un-whiten weights back to original space
        if self.whiten:
            W = W.matmul(C_inv_sqrt)

        # restore shape and assign
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.mask.data = M.to(dtype=self.layer.weight.dtype)
        if not disable_update:
            self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)

    def free(self):
        if DEBUG:
            self.inp1 = self.out1 = None
        self.H = self.C = None
        torch.cuda.empty_cache()


