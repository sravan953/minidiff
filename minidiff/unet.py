import math

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.t_proj = nn.Linear(128, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 3, padding="same")

    def forward(self, x: torch.Tensor, t_encoding: torch.Tensor) -> torch.Tensor:
        x = x + self.t_proj(t_encoding).unsqueeze(2).unsqueeze(3)
        x_out = nn.functional.relu(self.bn1(self.conv1(x)))
        x_out = nn.functional.relu(self.bn2(self.conv2(x_out)))
        x_out = x_out + self.residual(x)
        return x_out


class DownResBlock(ResBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.maxpool2d = nn.MaxPool2d(2)

    def forward(
        self, x: torch.Tensor, t_encoding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_out = super().forward(x, t_encoding)
        x_resampled = self.maxpool2d(x_out)
        return x_resampled, x_out


class UpResBlock(ResBlock):
    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, timestep_encoded: torch.Tensor
    ) -> torch.Tensor:
        x = nn.functional.interpolate(
            x, size=(skip.shape[2], skip.shape[3]), mode="bilinear"
        )
        x = torch.cat([x, skip], dim=1)
        x = x + self.t_proj(timestep_encoded).unsqueeze(2).unsqueeze(3)
        x_out = nn.functional.relu(self.bn1(self.conv1(x)))
        x_out = nn.functional.relu(self.bn2(self.conv2(x_out)))
        x_out = x_out + self.residual(x)
        return x_out


class UNet(nn.Module):
    def __init__(self, num_blocks: int, num_steps: int = 1000):
        super().__init__()

        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.register_buffer("sin_t_encoding", self.get_sin_t_encoding(dim=128))

        self.stem = nn.Conv2d(1, 32, 3, padding="same")
        self.down_blocks = nn.ModuleList(
            [DownResBlock(32 * i, 32 * (i + 1)) for i in range(1, self.num_blocks + 1)]
        )
        self.bottleneck = nn.ModuleList(
            [
                ResBlock(32 * (self.num_blocks + 1), 32 * (self.num_blocks + 3)),
                ResBlock(32 * (self.num_blocks + 3), 32 * (self.num_blocks + 1)),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpResBlock(2 * 32 * (i + 1), 32 * i)
                for i in range(self.num_blocks, 1, -1)
            ]
            + [UpResBlock(128, 64)]
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding="same"),
        )

    def get_sin_t_encoding(self, dim: int) -> torch.Tensor:
        position = torch.arange(1, self.num_steps + 1)[:, None]
        div_term = torch.exp(torch.arange(0, dim, 2) * -math.log(10000.0) / dim)

        embeddings = torch.zeros((self.num_steps, dim), dtype=torch.float32)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)

        return embeddings

    def forward(self, x: torch.Tensor, timestep: int) -> torch.Tensor:
        t_encoding = self.sin_t_encoding[timestep - 1]

        x = self.stem(x)
        all_skips = []
        for down_block in self.down_blocks:
            x, x_skip = down_block(x, t_encoding)
            all_skips.append(x_skip)
        all_skips.reverse()
        for bottleneck in self.bottleneck:
            x = bottleneck(x, t_encoding)
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, all_skips[i], t_encoding)
        x = self.final_conv(x)

        return x
