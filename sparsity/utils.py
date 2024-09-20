import torch

def replace_linear_with_(model, new_class, exclude=[], **kwargs):
    """replace linear layers with new_class in a model. It's a inplace operation."""
    for name, module in model.named_children():
        if module in exclude:
            continue
        if isinstance(module, torch.nn.Linear):
            new_linear = new_class(module.in_features, module.out_features, module.bias is not None, **kwargs)
            new_linear.weight.data = module.weight.data
            if module.bias is not None:
                new_linear.bias.data = module.bias.data
            new_linear.to(module.weight.device)
            setattr(model, name, new_linear)   
        else:
            replace_linear_with_(module, new_class, exclude)     
    return model