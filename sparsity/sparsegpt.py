import math
import time

import torch
import torch.nn as nn
import transformers

# -----------------------------------------------------------------------------
# SparseGPT‑Mahalanobis *minimal diff* edition
# -----------------------------------------------------------------------------
# ‑ Keeps all public methods / arguments identical to the original (ICML‑23)
# ‑ Adds an **output‑side Mahalanobis metric** through the Kronecker Hessian
#     H = (XXᵀ) ⊗ Σ⁻¹   with  Σ = YYᵀ / N  collected during add_batch().
# ‑ Update order, variable names, inner loops, and printed diagnostics mirror
#   the vanilla code as closely as possible, so you can diff line‑by‑line.
# -----------------------------------------------------------------------------

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class SparseGPT:

    # ------------------------------------------------------------
    # INITIALISATION (unchanged)
    # ------------------------------------------------------------
    def __init__(self, layer):
        self.layer = layer
        self.dev   = layer.weight.device

        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()

        self.rows    = W.size(0)   # d_out
        self.columns = W.size(1)   # d_in

        # input covariance (XXᵀ) and output covariance (YYᵀ)
        self.H     = torch.zeros((self.columns, self.columns), device=self.dev)
        self.Sigma = torch.zeros((self.rows,   self.rows),   device=self.dev)
        self.nsamples = 0

    # ------------------------------------------------------------
    # ACCUMULATE COVARIANCES (add_batch)
    # ------------------------------------------------------------
    def add_batch(self, inp, out, blocksize: int = 1024):
        if DEBUG:
            self.inp1, self.out1 = inp, out

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            out = out.unsqueeze(0)

        # flatten leading dims for linear / Conv1D to match original shapes
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)) and len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
            out = out.reshape((-1, out.shape[-1]))
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            inp = inp.t()   # (d_in,  N)
            out = out.t()   # (d_out, N)

        Nb = inp.size(1)
        self.H     *= self.nsamples / (self.nsamples + Nb)
        self.Sigma *= self.nsamples / (self.nsamples + Nb)
        self.nsamples += Nb

        scale = math.sqrt(2 / self.nsamples)
        inp, out = inp.float() * scale, out.float() * scale
        self.H     += inp.matmul(inp.t())
        self.Sigma += out.matmul(out.t())

    # ------------------------------------------------------------
    # PRUNE PASS (fasterprune) — structure identical to baseline
    # ------------------------------------------------------------
    def fasterprune(
        self,
        sparsity_ratio: float,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
        disable_update: bool = False,
    ):
        # --- flatten weights like the original --------------------------------
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        M = torch.ones_like(W)

        tic = time.time()

        # --------------------------- INPUT CURVATURE --------------------------
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        diag_idx = torch.arange(self.columns, device=self.dev)
        H[diag_idx, diag_idx] += percdamp * torch.mean(torch.diag(H))

        L   = torch.linalg.cholesky(H)              # L Lᵀ = H
        Ainv = torch.cholesky_inverse(L)            # (XXᵀ+λI)⁻¹
        R   = torch.linalg.cholesky(Ainv, upper=True)   # Rᵀ R = Ainv

        # cache useful pieces
        Ainv_diag      = torch.diag(Ainv)           # vector (d_in)
        R_diag         = torch.diag(R)              # sqrt(Ainv_diag)

        # --------------------------- OUTPUT CURVATURE -------------------------
        S = self.Sigma
        del self.Sigma
        row_idx = torch.arange(self.rows, device=self.dev)
        S[row_idx, row_idx] += percdamp * torch.mean(torch.diag(S))

        LS      = torch.linalg.cholesky(S)          # lower‑tri Σ^{1/2}
        Sinv    = torch.cholesky_inverse(LS)        # Σ⁻¹
        Sinv_d  = torch.diag(Sinv)                  # precisions per row

        # --------------------------------------------------------------------
        Losses = torch.zeros(self.rows, device=self.dev)
        mask_global = None

        for i1 in range(0, self.columns, blocksize):
            i2     = min(i1 + blocksize, self.columns)
            count  = i2 - i1

            W1 = W[:, i1:i2].clone()
            M1 = M[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)             # store Σ·err per column
            Losses1 = torch.zeros_like(W1)
            Rblock  = R[i1:i2, i1:i2]               # upper‑tri sqrt(Ainv) slice

            # importance scores ------------------------------------------------
            denom_block = Sinv_d[:, None] * Ainv_diag[i1:i2][None, :]
            score_block = (W1 ** 2) / denom_block
            if prunen == 0:
                if mask_global is not None:
                    mask1 = mask_global[:, i1:i2]
                else:
                    thresh = torch.sort(score_block.flatten())[0][int(score_block.numel() * sparsity_ratio)]
                    mask1  = score_block <= thresh
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool, device=self.dev)

            # structured N:M pruning -----------------------------------------
            if prunen != 0:
                for j in range(0, count, prunem):
                    sub = score_block[:, j:j+prunem]
                    k_small = torch.topk(sub, prunen, dim=1, largest=False)[1]
                    rows_i  = torch.arange(sub.size(0), device=self.dev).unsqueeze(1)
                    mask1[rows_i, j + k_small] = True

            for i in range(count):
                col_global = i1 + i
                w  = W1[:, i]
                m  = M1[:, i]
                d  = R_diag[col_global]             # == sqrt(Ainv[ii])

                # apply mask --------------------------------------------------
                q        = w.clone()
                q[mask1[:, i]] = 0
                m[mask1[:, i]] = 0
                Q1[:, i] = q
                M1[:, i] = m

                denom_row = Sinv_d * Ainv_diag[col_global]
                Losses1[:, i] = (w - q) ** 2 / denom_row

                if disable_update:
                    continue

                r_vec = w - q                          # removed weights (rows)
                u_vec = S.matmul(r_vec)                # Σ · r   (rows)

                # --- update current block  (columns i … count‑1) ----------
                a_row_block = Ainv[col_global, col_global:i2]
                W1[:, i:] -= u_vec.unsqueeze(1) * a_row_block.unsqueeze(0)

                Err1[:, i] = u_vec                     # cache for future cols

            if not disable_update:
                W[:, i1:i2] = Q1                       # commit pruned weights

            M[:, i1:i2] = M1
            Losses += torch.sum(Losses1, 1) / 2

            # propagate error to *future* columns ----------------------------
            if not disable_update and i2 < self.columns:
                A_future = Ainv[i1:i2, i2:]            # (count × remaining)
                W[:, i2:] -= Err1.matmul(A_future)

        torch.cuda.synchronize()
        print(f"time {time.time() - tic:.2f}s")
        print("error", torch.sum(Losses).item())

        # restore layout for Conv1D -----------------------------------------
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.layer.mask.data = M.to(dtype=self.layer.weight.dtype)
        if not disable_update:
            self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    # ------------------------------------------------------------
    # FREE
    # ------------------------------------------------------------
    def free(self):
        if DEBUG:
            self.inp1 = self.out1 = None
        self.H = self.Sigma = None
        torch.cuda.empty_cache()

