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
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows, self.columns = W.shape

        # initialize Hessian and covariance for whitening
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.C = torch.zeros_like(self.H)
        self.nsamples = 0

        # whitening switch and epsilon
        self.whiten = whiten
        self.eps = eps

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        # update Hessian H
        self.H *= self.nsamples / (self.nsamples + batch)
        scaled = math.sqrt(2 / (self.nsamples + batch)) * inp.float()
        self.H += scaled.matmul(scaled.t())

        # update covariance C
        self.C *= self.nsamples / (self.nsamples + batch)
        self.C += inp.float().matmul(inp.float().t())

        self.nsamples += batch

    def fasterprune(
        self, sparsity_ratio, prunen=0, prunem=0, blocksize=128,
        percdamp=.01, disable_update=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        M = torch.ones_like(W)

        tick = time.time()

        # apply whitening to H and W if enabled
        if self.whiten and self.nsamples > 0:
            # regularize and damp covariance for whitening
            base = self.C / self.nsamples
            damp_C = percdamp * torch.mean(torch.diag(base))
            C_reg = base + (self.eps + damp_C) * torch.eye(self.columns, device=self.dev)
            Lc = torch.linalg.cholesky(C_reg)
            C_inv_sqrt = torch.cholesky_inverse(Lc)
            C_sqrt = Lc
            H = C_inv_sqrt.matmul(self.H).matmul(C_inv_sqrt)
            W = W.matmul(C_sqrt)
        else:
            H = self.H

        dead = torch.diag(H) == 0
        H = H.clone()
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            M1 = M[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity_ratio)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1)

            for i in range(count):
                w = W1[:, i]
                m = M1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0
                m[mask1[:, i]] = 0

                Q1[:, i] = q
                M1[:, i] = m
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                if not disable_update:
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
            if not disable_update:
                W[:, i1:i2] = Q1
            M[:, i1:i2] = M1
            Losses += torch.sum(Losses1, 1) / 2
            if not disable_update:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        # un-whiten weights back if applied
        if self.whiten and self.nsamples > 0:
            W = W.matmul(C_inv_sqrt)

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.layer.mask.data = M.to(dtype=self.layer.weight.dtype)
        if not disable_update:
            self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.C = None
        torch.cuda.empty_cache()
