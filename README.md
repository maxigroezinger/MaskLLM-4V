# MaskLLM-4V
Unofficial re-implementation of the paper "MaskLLM: Learnable Semi-structured Sparsity for Large Language Models" for Vision Transformers -- ViTs, DiTs, etc.

![maskllm_framework](assets/framework.png)
![gumbel_softmax](assets/gumbel_softmax.png)

## TODO List

- [x] 2:4 Sparsity for Vision Transformers on ImageNet-1k 
- [] 2:4 Sparsity for Diffusion Transformers on ImageNet-1k
- [] TensorRT examples

## Results on ViTs

### [ViT-B/16 (augreg_in1k, 224x224)](https://huggingface.co/timm/vit_base_patch16_224.augreg_in1k)

|Method|Sparsity Pattern|Weight Update| Top-1 Acc. (%) |
|---|:---:|:---:|:---:|
| ViT-B/16 (in1k) | Dense | - | 79.15 |
||
|Magnitude| 2:4 | - | 65.92 |
|Wanda| 2:4 | - | 63.28 |
|SparseGPT| 2:4 | :heavy_check_mark: | 71.52 |
|SparseGPT w/o Update| 2:4 | - | 59.72 |
| **MaskLLM-4V** | **2:4** | - | **79.45** |

### [ViT-B/16 (augreg2_in21k_ft_in1k, 224x224)](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)

|Method|Sparsity Pattern|Weight Update| Top-1 Acc. (%) |
|---|:---:|:---:|:---:|
| ViT-B/16 (in21k_ft_in1k) | Dense | - | 85.10 |
||
|Magnitude| 2:4 | - | 53.91 |
|Wanda| 2:4 | - | 67.38 |
|SparseGPT| 2:4 | :heavy_check_mark: | 79.75 |
|SparseGPT w/o Update| 2:4 | - | 62.86 |
| **MaskLLM-4V** | **2:4** | - | 83.52 |


### 1. MaskLLM

#### Generate Mask Prior

First, we generate prior masks with SparseGPT. This prior mask will hugely accelerate the convergence speed of the MaskLLM. Also you can use magnitude pruning for the prior mask.

```bash 
python oneshot_pruning_timm.py --model vit_base_patch16_224.augreg_in1k  --pruner sparsegpt --save-model output/pruned/vit_base_patch16_224.augreg_in1k.sparsegpt24.pt
```

#### Train MaskLLM based on the Magnitude Prior
We took the hyperparameters from [this timm issue](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/discussions/1). 
```bash
bash scripts/maskllm_vit_base_patch16_224.augreg_in1k.magnitude24.sh
```

#### Evalulate MaskLLM
```bash
python timm_validate.py --model vit_base_patch16_224 --checkpoint CKPT_PATH --sparsity-mode maskllm
```

To perform MaskLLM on a different model or prior type, you can change the `--model` and `--checkpoint` arguments. For example, we produce masks for `vit_base_patch16_224.augreg2_in21k_ft_in1k` model with the following commands:

```bash
# For augreg2_in21k_ft_in1k
python oneshot_pruning_timm.py --model vit_base_patch16_224.augreg2_in21k_ft_in1k  --pruner sparsegpt --save-model output/pruned/vit_base_patch16_224.augreg2_in21k_ft_in1k.sparsegpt24.pt 

bash scripts/maskllm_vit_base_patch16_224.augreg2_in21k_ft_in1k.magnitude24.sh

python timm_validate.py --model vit_base_patch16_224 --checkpoint CKPT_PATH --sparsity-mode maskllm
```

### 2. Dense

<details>
<summary>Detailed Instructions</summary>

#### ImageNet-1K:
```bash
# ImageNet-1k
python timm_validate.py --model vit_base_patch16_224.augreg_in1k  --pretrained
```
```json
{
    "model": "vit_base_patch16_224.augreg_in1k",
    "top1": 79.158,
    "top1_err": 20.842,
    "top5": 94.088,
    "top5_err": 5.912,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```

#### ImageNet-21k-ft-1k:
```bash
python timm_validate.py --model vit_base_patch16_224.augreg2_in21k_ft_in1k  --pretrained
```
```json
{
    "model": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "top1": 85.096,
    "top1_err": 14.904,
    "top5": 97.53,
    "top5_err": 2.47,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```

</details>


### 3. Magnitude Pruning
<details>
<summary>Detailed Instructions</summary>

#### Magnitude - ImageNet-1K
```bash
# ImageNet-1k
python oneshot_pruning_timm.py --model vit_base_patch16_224.augreg_in1k --pruner magnitude --save-model output/pruned/vit_base_patch16_224.augreg_in1k.magnitude24.pt

# Eval
python timm_validate.py --model vit_base_patch16_224 --checkpoint output/pruned/vit_base_patch16_224.augreg_in1k.magnitude24.pt --sparsity-mode sparse
```
```json
{
    "model": "vit_base_patch16_224",
    "top1": 65.92,
    "top1_err": 34.08,
    "top5": 86.058,
    "top5_err": 13.942,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```

#### Magnitude - ImageNet-21k-ft-1k
```bash
python oneshot_pruning_timm.py --model vit_base_patch16_224.augreg2_in21k_ft_in1k --pruner magnitude --save-model output/pruned/vit_base_patch16_224.augreg2_in21k_ft_in1k.magnitude24.pt

# Eval
python timm_validate.py --model vit_base_patch16_224 --checkpoint output/pruned/vit_base_patch16_224.augreg2_in21k_ft_in1k.magnitude24.pt --sparsity-mode sparse
```
```json
{
    "model": "vit_base_patch16_224",
    "top1": 53.906,
    "top1_err": 46.094,
    "top5": 77.358,
    "top5_err": 22.642,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```

</details>

### 4. Wanda
<details>
<summary>Detailed Instructions</summary>

#### Wanda - ImageNet-1K
```bash
python oneshot_pruning_timm.py --model vit_base_patch16_224.augreg_in1k  --pruner wanda --save-model output/pruned/vit_base_patch16_224.augreg_in1k.wanda24.pt

# Eval
python timm_validate.py --model vit_base_patch16_224 --checkpoint output/pruned/vit_base_patch16_224.augreg_in1k.wanda24.pt --sparsity-mode sparse
```
```json
{
    "model": "vit_base_patch16_224",
    "top1": 63.282,
    "top1_err": 36.718,
    "top5": 85.574,
    "top5_err": 14.426,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```


#### Wanda - ImageNet-21k-ft-1k
```bash
python oneshot_pruning_timm.py --model vit_base_patch16_224.augreg2_in21k_ft_in1k  --pruner wanda --save-model output/pruned/vit_base_patch16_224.augreg2_in21k_ft_in1k.wanda24.pt

# Eval
python timm_validate.py --model vit_base_patch16_224 --checkpoint output/pruned/vit_base_patch16_224.augreg2_in21k_ft_in1k.wanda24.pt --sparsity-mode sparse
```
```json
{
    "model": "vit_base_patch16_224",
    "top1": 67.378,
    "top1_err": 32.622,
    "top5": 88.7,
    "top5_err": 11.3,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```



</details>

### 5. SparseGPT

<details>
<summary>Detailed Instructions</summary>


#### SparseGPT - ImageNet-1K without update
```bash
python oneshot_pruning_timm.py --model vit_base_patch16_224.augreg_in1k --pruner sparsegpt --save-model output/pruned/vit_base_patch16_224.augreg_in1k.sparsegpt24.pt

# Eval
python timm_validate.py --model vit_base_patch16_224 --checkpoint output/pruned/vit_base_patch16_224.augreg_in1k.sparsegpt24.pt --sparsity-mode sparse
```
```json
{
    "model": "vit_base_patch16_224",
    "top1": 59.728,
    "top1_err": 40.272,
    "top5": 82.326,
    "top5_err": 17.674,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```

#### SparseGPT - ImageNet-1K with update

```bash
python oneshot_pruning_timm.py --model vit_base_patch16_224.augreg_in1k --pruner sparsegpt --save-model output/pruned/vit_base_patch16_224.augreg_in1k.sparsegpt24_updated.pt --enable-update

# Eval
python timm_validate.py --model vit_base_patch16_224 --checkpoint output/pruned/vit_base_patch16_224.augreg_in1k.sparsegpt24_updated.pt --sparsity-mode sparse
```
```json
{
    "model": "vit_base_patch16_224",
    "top1": 71.52,
    "top1_err": 28.48,
    "top5": 90.246,
    "top5_err": 9.754,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```


#### SparseGPT - ImageNet-21k-ft-1k without update

```bash
python oneshot_pruning_timm.py --model vit_base_patch16_224.augreg2_in21k_ft_in1k --pruner sparsegpt --save-model output/pruned/vit_base_patch16_224.augreg2_in21k_ft_in1k.sparsegpt24.pt

# Eval
python timm_validate.py --model vit_base_patch16_224 --checkpoint output/pruned/vit_base_patch16_224.augreg2_in21k_ft_in1k.sparsegpt24.pt --sparsity-mode sparse
```

```json
{
    "model": "vit_base_patch16_224",
    "top1": 62.858,
    "top1_err": 37.142,
    "top5": 85.4,
    "top5_err": 14.6,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```


#### SparseGPT - ImageNet-21k-ft-1k with update
```bash
python oneshot_pruning_timm.py --model vit_base_patch16_224.augreg2_in21k_ft_in1k --pruner sparsegpt --save-model output/pruned/vit_base_patch16_224.augreg2_in21k_ft_in1k.sparsegpt24_updated.pt --enable-update

# Eval
python timm_validate.py --model vit_base_patch16_224 --checkpoint output/pruned/vit_base_patch16_224.augreg2_in21k_ft_in1k.sparsegpt24_updated.pt --sparsity-mode sparse
```
```json
{
    "model": "vit_base_patch16_224",
    "top1": 79.754,
    "top1_err": 20.246,
    "top5": 95.51,
    "top5_err": 4.49,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```


</details>

## Acknowledgement

This project is based on the following repositories:

- [NVlabs/MaskLLM]()
- [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- [locuslab/wanda](https://github.com/locuslab/wanda)
- [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt)

## BibTeX

If you find this repository helpful, please consider citing the following paper.
```bibtex
```
