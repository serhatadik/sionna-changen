import torch
import torch.nn as nn

class BasicUNet1D(nn.Module):
    """A minimal UNet implementation for 1D signals."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = nn.ModuleList(
            [
                nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = nn.ModuleList(
            [
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
                nn.Conv1d(64, 32, kernel_size=5, padding=2),
                nn.Conv1d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = nn.SiLU()  # The activation function
        self.downscale = nn.MaxPool1d(2)
        self.upscale = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x