import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dil=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dil, dilation=dil),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.down1 = ConvBlock(2, ch) 
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(ch, ch*2, dil=2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ConvBlock(ch*2, ch*4, dil=4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck с расширенным полем зрения
        self.bottleneck = ConvBlock(ch*4, ch*8, dil=8)
        
        self.up3 = nn.ConvTranspose2d(ch*8, ch*4, 2, 2)
        self.dec3 = ConvBlock(ch*8, ch*4)
        self.up2 = nn.ConvTranspose2d(ch*4, ch*2, 2, 2)
        self.dec2 = ConvBlock(ch*4, ch*2)
        self.up1 = nn.ConvTranspose2d(ch*2, ch, 2, 2)
        self.dec1 = ConvBlock(ch*2, ch)
        self.final = nn.Conv2d(ch, 2, 1)

    def forward(self, x):
        identity = x
        h, w = x.shape[2], x.shape[3]
        pad_h, pad_w = (16 - h % 16) % 16, (16 - w % 16) % 16
        x = F.pad(x, (0, pad_w, 0, pad_h))

        d1 = self.down1(x)
        p1 = self.pool1(d1); d2 = self.down2(p1)
        p2 = self.pool2(d2); d3 = self.down3(p2)
        p3 = self.pool3(d3); bn = self.bottleneck(p3)

        # Добавляем немного вариативности в скрытое пространство
        if self.training:
            bn = bn + torch.randn_like(bn) * 0.01

        u3 = self.up3(bn); u3 = F.interpolate(u3, size=(d3.shape[2], d3.shape[3]))
        m3 = self.dec3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(m3); u2 = F.interpolate(u2, size=(d2.shape[2], d2.shape[3]))
        m2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(m2); u1 = F.interpolate(u1, size=(d1.shape[2], d1.shape[3]))
        m1 = self.dec1(torch.cat([u1, d1], dim=1))
        
        out = self.final(m1)[:, :, :h, :w]
        return out + identity