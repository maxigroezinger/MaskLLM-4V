import torch 
import torch.nn as nn 

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion_transformers.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from diffusion_transformers.download import find_model
from diffusion_transformers.models import DiT_models
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import argparse
import os

import sparsity
from typing import Any, Dict, List, Optional, Union
from sparsity.sparsegpt import SparseGPT


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if isinstance(module, tuple(layers)):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def prune_magnitude(model, device=torch.device("cuda:0"), prune_n=0, prune_m=0, sparsity_ratio=0.0, reverse=False):
    for name, layer in model.blocks.named_modules():
        if isinstance(layer, sparsity.maskllm.MaskedLinearFrozen):
            W = layer.weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)
            #W[W_mask] = 0
            if reverse:
                layer.mask.data = W_mask
            else:
                layer.mask.data = ~W_mask
            print(f"Layer {name} Sparsity: {torch.sum(W_mask).item()/W.numel()}")

@torch.no_grad()
def prune_wanda(model, vae, loader, nsamples=128, batch_size=1, device=torch.device("cuda:0"), prune_n=0, prune_m=0, sparsity_ratio=0.0):
    class InputOutputHook():
        def __init__(self, layer):
            self.layer = layer
            self.columns = layer.weight.data.shape[1]
            self.dev = self.layer.weight.device

            self.hook = layer.register_forward_hook(self.hook_fn)
            self.scaler_row = torch.zeros((self.columns), device=self.dev)
            self.nsamples = 0
        
        @torch.no_grad()
        def hook_fn(self, module, input, output):
            inp = input[0]
            if isinstance(self.layer, nn.Linear):
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()

            self.scaler_row *= self.nsamples / (self.nsamples + inp.shape[0])
            self.nsamples += inp.shape[0]
            self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        
        def close(self):
            self.hook.remove()

    nbatches = nsamples // batch_size

    layer_hook_pairs = []
    for name, layer in model.blocks.named_modules():
        if isinstance(layer, sparsity.maskllm.MaskedLinearFrozen):
            hook = InputOutputHook(layer)
            layer_hook_pairs.append((name, layer, hook))

    for i, (x, y) in enumerate(loader):
        if i == nbatches: break
        print(f"Batch {i+1}/{nbatches}")
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            inputs_dict = {
                "x": x,
                "t": t,
                "y": y,
            }
            _ = model(**inputs_dict)

    hook.close()
    for name, layer, hook in layer_hook_pairs:
        W_metric = torch.abs(layer.weight.data) * torch.sqrt(hook.scaler_row.reshape((1,-1)))
        W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
        # structured n:m sparsity
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:,ii:(ii+prune_m)].float()
                W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        layer.mask.data = ~W_mask
        print(f"Layer {name} Sparsity: {torch.sum(W_mask).item()/W_mask.numel()}")
        
@torch.no_grad()
def prune_sparsegpt(model, vae, loader, nsamples=128, batch_size=1, device=torch.device("cuda:0"), prune_n=0, prune_m=0, sparsity_ratio=0.0, disable_update=True):
    # Initialize input caches
    x_cache = [] #latent variables
    c_cache = [] 
    cache_index = [0]

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x, c):
            if cache_index[0] < nsamples:
                x_cache.append(x.float())
                c_cache.append(c.float())
                cache_index[0] += 1
            raise ValueError  # To stop the forward pass and capture the inputs
        
    # Replace the first transformer block with the Catcher
    model.blocks[0] = Catcher(model.blocks[0])
    
    nbatches = nsamples // batch_size
    for i, (x, y) in enumerate(loader):
        if i == nbatches: break
        print(f"Batch {i+1}/{nbatches}")
        x = x.to(device)
        y = y.to(device)
    
        try:
            # Process the prompt through the pipeline
            with torch.no_grad():
                #([1, 3, 256, 256])
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                #([1, 4, 32, 32])
                #y Size([1])
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                #Size([1])
                inputs_dict = {
                    "x": x,
                    "t": t,
                    "y": y,
                }
                _ = model(**inputs_dict)
        except ValueError:
            pass

    # Restore the original transformer block
    model.blocks[0] = model.blocks[0].module

    x_outs = []

    # Loop through transformer blocks and perform operations
    for i, block in enumerate(model.blocks):
        subset = find_layers(block, layers=[sparsity.maskllm.MaskedLinearFrozen])
        gpts = {name: SparseGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        # Register hooks
        handles = [subset[name].register_forward_hook(add_batch(name)) for name in gpts]

        try:
            # Process inputs through the layer
            for j in range(nsamples):
                x = x_cache[j]
                c = c_cache[j]
                block(
                    x=x,
                    c=c,
                )
        except Exception as e:
            raise e

        for h in handles:
            h.remove()

        # Pruning
        try:
            for name in gpts:
                sparsity_ratio = sparsity_ratio
                gpts[name].fasterprune(sparsity_ratio, prunen=prune_n, prunem=prune_m, blocksize=128, disable_update=disable_update)
                gpts[name].free()
        except Exception as e:
            raise e

        try:
            # Process inputs again through the layer
            for j in range(nsamples):
                x = x_cache[j]
                c = c_cache[j]

                x=block(
                    x=x,
                    c=c,
                )
                x_outs.append(x.float())

        except Exception as e:
            raise e

        x_cache, x_outs = x_outs, x_cache

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--data-path", type=str, default='data/imagenet/train') 
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--pruner", type=str, default='dense', choices=['dense', 'magnitude', 'random', 'sparsegpt', 'wanda'])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--enable-update", action='store_true', default=False)
    args = parser.parse_args()

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    model = sparsity.utils.replace_linear_with_(model, sparsity.maskllm.MaskedLinearFrozen, exclude=[model.final_layer.linear])
    model.to(device)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
    )
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)


    if args.pruner == 'dense':
        pass
    elif args.pruner == 'magnitude':
        prune_magnitude(model, device=device, prune_n=2, prune_m=4)
    elif args.pruner == 'wanda':
        prune_wanda(model, vae, loader=loader, nsamples=args.nsamples, batch_size=1, device=device, prune_n=2, prune_m=4)
    elif args.pruner == 'sparsegpt':
        prune_sparsegpt(model, vae, loader=loader, nsamples=args.nsamples, batch_size=1, device=device, prune_n=2, prune_m=4, disable_update=not args.enable_update)
    else:
        raise NotImplementedError

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    os.makedirs("output/samples_from_pruned_dits", exist_ok=True)
    save_image(samples,f"output/samples_from_pruned_dits/{args.model.replace('-', '_').replace('/', '_')}_{args.pruner}.png", nrow=4, normalize=True, value_range=(-1, 1))

    # Check sparsiy
    for name, m in model.named_modules():
        if hasattr(m, 'mask'):
            print(f"Layer {name} Sparsity: {1 - torch.sum(m.mask).item()/m.mask.numel()}")

    if args.save_model is not None:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(model.state_dict(), args.save_model)