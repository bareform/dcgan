import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(
            self,
            img_size: tuple[int, int, int],
            in_channels: list[int],
            latent_dim: int=100
        ) -> None:
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        layers = []
        layers.extend([
            nn.ConvTranspose2d(
                in_channels=self.latent_dim,
                out_channels=self.in_channels[0],
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.ReLU(inplace=True)
        ])
        for i in range(len(self.in_channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(self.in_channels[i + 1]),
                nn.ReLU(inplace=True)
            ])
        layers.extend([
            nn.ConvTranspose2d(
                in_channels=self.in_channels[-1],
                out_channels=self.img_size[0],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        ])
        self.generator_layers = nn.Sequential(*layers)

    def forward(self, input: torch.tensor) -> torch.tensor:
        out = self.generator_layers(input)
        return out
