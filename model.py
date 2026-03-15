import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, channels, 9, padding=4)
        self.conv2 = nn.Conv1d(channels, channels, 9, padding=4)

        self.act = nn.PReLU()

    def forward(self, x):

        r = x

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)

        return x + r


class AudioUpscaler(nn.Module):

    def __init__(self):

        super().__init__()

        self.entry = nn.Conv1d(1, 32, 9, padding=4)

        self.blocks = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.exit = nn.Conv1d(32, 1, 9, padding=4)

    def forward(self, x):

        x = self.entry(x)
        x = self.blocks(x)
        x = self.exit(x)

        return x