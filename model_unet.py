import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.down1 = ConvBlock(1, ch)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ConvBlock(ch, ch*2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = ConvBlock(ch*2, ch*4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(ch*4, ch*8)

        self.up3 = nn.ConvTranspose2d(ch*8, ch*4, 2, 2)
        self.dec3 = ConvBlock(ch*8, ch*4)

        self.up2 = nn.ConvTranspose2d(ch*4, ch*2, 2, 2)
        self.dec2 = ConvBlock(ch*4, ch*2)

        self.up1 = nn.ConvTranspose2d(ch*2, ch, 2, 2)
        self.dec1 = ConvBlock(ch*2, ch)

        self.final = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        def pad_to_match(source, target):
            diffY = target.size()[2] - source.size()[2]
            diffX = target.size()[3] - source.size()[3]
            import torch.nn.functional as F
            return F.pad(source, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])

        # Down
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        # Bottom
        bn = self.bottleneck(p3)

        # Up
        u3 = self.up3(bn)
        u3 = pad_to_match(u3, d3) 
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.dec3(u3)

        u2 = self.up2(u3)
        u2 = pad_to_match(u2, d2) 
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = pad_to_match(u1, d1)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1)

        return self.final(u1)