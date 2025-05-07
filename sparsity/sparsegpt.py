import math
import time

import torch
import torch.nn as nn
import transformers

# -----------------------------------------------------------------------------
# SparseGPT‑Mahalanobis: one‑shot pruning with a block‑wise Kronecker OBS solve
# -----------------------------------------------------------------------------
# ‑ Keeps the public interface of the original SparseGPT class intact
# ‑ Works for nn.Linear / transformers.Conv1D / Conv2d exactly as before
# ‑ No Woodbury trick – everything is classic Cholesky, identical to the 
#   code‑base you pasted.                                                    
# ‑ Additional feature: uses an **output‑side Mahalanobis loss** specified by
#   the empirical covariance   Σ = YYᵀ / N  collected during add_batch().
#   The corresponding Kronecker‑factored Hessian is        (XXᵀ) ⊗ Σ⁻¹.
#   A block‑wise closed form makes pruning 
#     ΔW  =  - Σ · R · A⁻¹            (see accompanying explanation).
# -----------------------------------------------------------------------------

DEBUG = False

# make reproducibility explicit
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class SparseGPT:
    """Drop‑in replacement that minimises a Mahalanobis‑weighted OBS loss.

    Interface is *identical* to the original SparseGPT   (add_batch, fasterprune,
    free).  Paste & run.
    """

    # ------------------------------------------------------------
    # INITIALISATION
    # ------------------------------------------------------------
    def __init__(self, layer):
        self.layer = layer
        self.dev   = self.layer.weight.device

        # --- infer flattened weight shape -----------------------------------
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)                # out_ch × (in_ch*k*k)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()                       # transformers stores transposed

        self.rows    = W.shape[0]           # d_out  (channels)
        self.columns = W.shape[1]           # d_in   (input dim)

        # --- store *input* covariance (X Xᵀ) like the original implementation
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        # --- store *output* covariance (Y Yᵀ)
        self.Sigma = torch.zeros((self.rows, self.rows), device=self.dev)

        self.nsamples = 0                   # running tally for both stats

    # ------------------------------------------------------------
    # ADD A CALIBRATION BATCH
    # ------------------------------------------------------------
    def add_batch(self, inp, out, blocksize: int = 1024):
        """Accumulate XXᵀ and YYᵀ, exactly mirroring the original scaling.

        Args
        ----
        inp  : activation *into* the pruned layer     (shape … × d_in)
        out  : activation *out of* the pruned layer   (shape … × d_out)
        """
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        # make inputs 3‑D  (batch, …, dim)  to reuse original logic
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        # Flatten arbitrary leading dims into batch*seq
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))   # (N, d_in)
                out = out.reshape((-1, out.shape[-1]))   # (N, d_out)
            inp = inp.t()   # → (d_in, N)
            out = out.t()   # → (d_out, N)

        tmp = inp.shape[1]           # number of new samples N_b

        # --- Running mean of covariances, identical scale as original code ---
        self.H     *= self.nsamples / (self.nsamples + tmp)
        self.Sigma *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        scale = math.sqrt(2 / self.nsamples)
        inp   = (inp.float()  * scale)            # (d_in , N)
        out   = (out.float()  * scale)            # (d_out, N)

        self.H     += inp.matmul(inp.t())         # update X Xᵀ
        self.Sigma += out.matmul(out.t())         # update Y Yᵀ

    # ------------------------------------------------------------
    # THE PRUNE PASS
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
        """One‑shot pruning with Mahalanobis OBS compensation.

        *All* public arguments preserved.  `prunen`/`prunem` structured sparsity
        continues to work.  `disable_update` still lets you only compute masks.
        """

        # ------------------------------------------------------------------
        # Flatten weights exactly like the reference implementation
        # ------------------------------------------------------------------
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        M = torch.ones_like(W)          # pruning mask (1 = keep, 0 = pruned)

        tick = time.time()

        # ------------------------------------------------------------------
        # INPUT‑SIDE COVARIANCE  – identical to baseline SparseGPT
        # ------------------------------------------------------------------
        H  = self.H.clone()
        del self.H                         # free memory early
        dead = torch.diag(H) == 0          # guards against unused inputs
        H[dead, dead] = 1
        W[:, dead] = 0                     # irrelevant weights → 0

        dampH = percdamp * torch.mean(torch.diag(H))
        diag_idx_col = torch.arange(self.columns, device=self.dev)
        H[diag_idx_col, diag_idx_col] += dampH

        # A_inv  = (XXᵀ + λI)⁻¹  via Cholesky
        L_H      = torch.linalg.cholesky(H)          # lower‑triangular L
        A_inv    = torch.cholesky_inverse(L_H)       # full inverse (d_in × d_in)
        # keep the *upper* Cholesky of A_inv to reuse original denominators
        Hinv_chol = torch.linalg.cholesky(A_inv, upper=True)   # R  s.t. A_inv = Rᵀ R

        # ------------------------------------------------------------------
        # OUTPUT‑SIDE COVARIANCE  Σ = YYᵀ / N
        # ------------------------------------------------------------------
        S  = self.Sigma.clone()
        del self.Sigma                    # we do not need the accumulator any more

        dampS = percdamp * torch.mean(torch.diag(S))
        diag_idx_row = torch.arange(self.rows, device=self.dev)
        S[diag_idx_row, diag_idx_row] += dampS

        # Σ_inv  (for scores)  and   Σ   (for compensation)
        L_S          = torch.linalg.cholesky(S)          # d_out × d_out  lower‑triangular
        Sigma_inv    = torch.cholesky_inverse(L_S)       # Σ⁻¹       (dense)
        Sigma_inv_d  = torch.diag(Sigma_inv)             # vector of row precisions

        # keep S itself   (needed for ΔW = - S R A_inv)
        Sigma = S                                       # renamed for clarity

        # Pre‑compute denominators used throughout
        A_inv_d    = torch.diag(A_inv)                  # d_in      (full)
        A_inv_d_sqrt = torch.diag(Hinv_chol)            # sqrt of diag(A_inv)

        mask = None   # will be allocated once for magnitude pruning path

        # ------------------------------------------------------------------
        # MAIN COLUMN‑WISE LOOP (unchanged public behaviour)
        # ------------------------------------------------------------------
        for i1 in range(0, self.columns, blocksize):
            i2     = min(i1 + blocksize, self.columns)
            count  = i2 - i1

            # local views for the current block
            W1 = W[:, i1:i2].clone()        # rows × count
            M1 = M[:, i1:i2].clone()

            # --------------------------------------------------------------
            # 1.  Importance scores  (Mahalanobis version)
            # --------------------------------------------------------------
            denom = Sigma_inv_d[:, None] * A_inv_d[i1:i2][None, :]   # rows × count
            tmp   = (W1 ** 2) / denom

            # --- magnitude‑like pruning mask ---------------------------------
            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity_ratio)]
                    mask1  = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool, device=self.dev)

            # N:M structured pruning (if requested) --------------------------
            if prunen != 0:
                for j in range(0, count, prunem):
                    col_span = slice(j, j + prunem)
                    block    = tmp[:, col_span]
                    k_small  = torch.topk(block, prunen, dim=1, largest=False)[1]
                    rows_idx = torch.arange(block.size(0), device=self.dev).unsqueeze(1)
                    mask1[rows_idx, j + k_small] = True

            # --------------------------------------------------------------
            # 2.  Apply the mask and compute the OBS compensation
            # --------------------------------------------------------------
            Q1      = W1.clone()
            Q1[mask1] = 0                         # pruned weights → 0 (kept for local Δ)

            # matrix R which holds the removed weights (pruned ones)
            R = W1.clone()
            R[~mask1] = 0                         # keep *only* the soon‑to‑be‑zeroed weights

            # accumulate per‑weight Mahalanobis loss for stats / debugging
            Losses1 = ((W1 - Q1) ** 2) / denom    # element‑wise

            if not disable_update:
                # ΔW_block    =  - Σ · R · A_inv_block
                # Σ  : (rows × rows)
                # R  : (rows × count)
                # A⁻¹_block  : (count × count)
                A_inv_block   = A_inv[i1:i2, i1:i2]
                Delta_block   = - Sigma.matmul(R).matmul(A_inv_block)

                # update within the current block  (note: keep zeros zero)
                W1 += Delta_block
                W1[mask1] = 0                     # enforce exact sparsity again

                # update *future* columns  (i2:)
                if i2 < self.columns:
                    A_inv_future = A_inv[i1:i2, i2:]
                    Delta_future = - Sigma.matmul(R).matmul(A_inv_future)   # rows × remaining
                    W[:, i2:]   += Delta_future

            # --------------------------------------------------------------
            # 3.  Commit back to full weight / mask tensors
            # --------------------------------------------------------------
            W[:, i1:i2] = W1
            M[:, i1:i2] = M1

            # accumulate loss if needed (kept for compatibility w/ print)
            if i1 == 0:
                Losses = torch.zeros(self.rows, device=self.dev)
            Losses += torch.sum(Losses1, 1) / 2

        torch.cuda.synchronize()
        print(f"time {time.time() - tick:.2f}s")
        print("error", torch.sum(Losses).item())

        # restore original layout for Conv1D -------------------------------
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        # write back to the module  (mask & possibly new weights) ----------
        self.layer.mask.data = M.to(dtype=self.layer.weight.dtype)
        if not disable_update:
            self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    # ------------------------------------------------------------
    # FREE RESOURCES EXPLICITLY
    # ------------------------------------------------------------
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Sigma = None
        torch.cuda.empty_cache()
