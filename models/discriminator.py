import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(
            self,
            img_size: tuple[int, int, int],
            in_channels: list[int],
            use_spectral_norm: bool=False
        ) -> None:
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        layers = []
        layers.extend([
            spectral_norm(nn.Conv2d(
                in_channels=self.img_size[0],
                out_channels=self.in_channels[0],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )) if use_spectral_norm else
            nn.Conv2d(
                in_channels=self.img_size[0],
                out_channels=self.in_channels[0],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        for i in range(len(self.in_channels) - 1):
            layers.extend([
                spectral_norm(nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )) if use_spectral_norm else
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        layers.append(
            spectral_norm(nn.Conv2d(
                in_channels=self.in_channels[-1],
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            )) if use_spectral_norm else
            nn.Conv2d(
                in_channels=self.in_channels[-1],
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ) 
        )
        self.discriminator_layers = nn.Sequential(*layers)

    def forward(self, input: torch.tensor) -> torch.tensor:
        out = self.discriminator_layers(input)
        return out
