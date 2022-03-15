import torch


def forzen_param(model):
    for name, param in model.named_parameters():
        if 'vid_proj' in name or 'txt_proj' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return True