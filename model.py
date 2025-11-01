import torch
import torch.nn as nn
import torch.nn.functional as F

# Customized U-Net-style generator (skip connections + other modifications)
class Generator(nn.Module):
    def __init__(self, in_channels=3, base_filters=64):
        super().__init__()
        # encoder blocks
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters*2)
        self.enc3 = self._conv_block(base_filters*2, base_filters*4)
        self.enc4 = self._conv_block(base_filters*4, base_filters*8)

        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = self._conv_block(base_filters*8, base_filters*8)

        # decoder (uses ConvTranspose for upsampling)
        self.up1 = nn.ConvTranspose2d(base_filters*8, base_filters*8, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_filters*8 + base_filters*8, base_filters*4)

        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*4, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_filters*4 + base_filters*4, base_filters*2)

        self.up3 = nn.ConvTranspose2d(base_filters*2, base_filters*2, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_filters*2 + base_filters*2, base_filters)

        # final conv
        self.final = nn.Conv2d(base_filters, in_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        d1 = self.enc1(x)         
        d2 = self.enc2(self.pool(d1))
        d3 = self.enc3(self.pool(d2))
        d4 = self.enc4(self.pool(d3))

        b = self.bottleneck(self.pool(d4))

        u1 = self.up1(b)
        u1 = torch.cat([u1, d4], dim=1)
        u1 = self.dec1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.dec2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.dec3(u3)

        # up to d1 size using interpolation if needed
        out = self.final(u3)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return torch.tanh(out)


# PatchGAN discriminator (6-channel input: condition + image)
class Discriminator(nn.Module):
    def __init__(self, in_channels=6, base_filters=64):
        super().__init__()
        def conv(ic, oc, stride=2, use_bn=True):
            layers = [nn.Conv2d(ic, oc, kernel_size=4, stride=stride, padding=1, bias=False)]
            if use_bn:
                layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv(in_channels, base_filters, use_bn=False),
            conv(base_filters, base_filters*2),
            conv(base_filters*2, base_filters*4),
            conv(base_filters*4, base_filters*8, stride=1),
            nn.Conv2d(base_filters*8, 1, kernel_size=4, stride=1, padding=1)  # logits
        )

    def forward(self, x):
        return self.model(x)