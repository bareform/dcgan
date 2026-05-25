from models import Generator

import torch
import torch.nn as nn

def weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

def gen(generator: Generator, z: torch.tensor, device: torch.device) -> torch.tensor:
    generator.eval()
    with torch.no_grad():
        z = z.to(device)
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        if z.size(1) != generator.latent_dim:
            z = z[:, :generator.latent_dim]
        images = generator(z)
        images = images / 2 + 0.5
        images = images * 255
    generator.train()
    return images
