# MaskLLM-4V
Re-implementation of "MaskLLM: Learnable Semi-structured Sparsity for Large Language Models" for Vision Models -- ViTs, DiTs, etc.

## 1. Results Overview

### [ViT-B/16 (Augreg_in21k_ft_in1k, 224x224)](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k)

|Method|Sparse Pattern|Weight Update| Top-1 Acc.|
|---|:---:|:---:|:---:|
|Dense ViT-B/16 | 2:4 | - | 84.532 |
|Magnitude| 2:4 | - | 38.248 |
|Wanda| 2:4 | - | 55.826 |
|SparseGPT| 2:4 | :heavy_check_mark: | 74.968 |
|SparseGPT w/o Update| 2:4 | - | 51.154 |
| MaskLLM-4V | 2:4 | - |  |



## 2. MaskLLM for Vision Transformers






## 3. Oneshot Pruning

### Dense
No pruning will be perfomed, but additional .mask will be created for the model.
```bash
# Pruning (Dummy for Dense)
python oneshot_pruning_timm.py  --pruner dense --save-model outputs/pruned/vit_base_patch16_224.augreg_in21k_ft_in1k.dense.pt

# Eval
python timm_validate.py --model vit_base_patch16_224 --checkpoint outputs/pruned/vit_base_patch16_224.augreg_in21k_ft_in1k.dense.pt --sparse
```

Output:
```json
{
    "model": "vit_base_patch16_224",
    "top1": 84.532,
    "top1_err": 15.468,
    "top5": 97.294,
    "top5_err": 2.706,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```


### Magnitude Pruning

```bash
# Pruning
python oneshot_pruning_timm.py  --pruner magnitude --save-model outputs/pruned/vit_base_patch16_224.augreg_in21k_ft_in1k.magnitude24.pt

# Eval
python timm_validate.py --model vit_base_patch16_224 --checkpoint outputs/pruned/vit_base_patch16_224.augreg_in21k_ft_in1k.magnitude24.pt --sparse
```

Output:
```json
{
    "model": "vit_base_patch16_224",
    "top1": 38.248,
    "top1_err": 61.752,
    "top5": 62.036,
    "top5_err": 37.964,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```

### Wanda
```bash
# Pruning
python oneshot_pruning_timm.py  --pruner wanda --save-model outputs/pruned/vit_base_patch16_224.augreg_in21k_ft_in1k.wanda24.pt

# Eval
python timm_validate.py --model vit_base_patch16_224 --checkpoint outputs/pruned/vit_base_patch16_224.augreg_in21k_ft_in1k.wanda24.pt --sparse
```

Ouput:
{
    "model": "vit_base_patch16_224",
    "top1": 55.826,
    "top1_err": 44.174,
    "top5": 80.064,
    "top5_err": 19.936,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}


### SparseGPT
```bash
# Pruning
python oneshot_pruning_timm.py  --pruner sparsegpt --save-model outputs/pruned/vit_base_patch16_224.augreg_in21k_ft_in1k.sparsegpt24.pt

# SparseGPT without weight update
python oneshot_pruning_timm.py  --pruner sparsegpt --save-model outputs/pruned/vit_base_patch16_224.augreg_in21k_ft_in1k.sparsegpt24.pt --disable-update

# Eval
python timm_validate.py --model vit_base_patch16_224 --checkpoint outputs/pruned/vit_base_patch16_224.augreg_in21k_ft_in1k.sparsegpt24.pt --sparse
```

Output:
```json
{
    "model": "vit_base_patch16_224",
    "top1": 74.968,
    "top1_err": 25.032,
    "top5": 93.386,
    "top5_err": 6.614,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```

Without update:
```json
{
    "model": "vit_base_patch16_224",
    "top1": 51.154,
    "top1_err": 48.846,
    "top5": 75.614,
    "top5_err": 24.386,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```