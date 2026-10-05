import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        groups = min(8, out_ch)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class TransformerBottleneck(nn.Module):

    def __init__(
        self,
        in_ch=512,
        embed_dim=384,
        depth=4,
        heads=6,
        base_hw=16
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            1
        )

        # Trainable parameter exists before optimizer creation.
        self.pos2d = nn.Parameter(
            torch.zeros(
                1,
                embed_dim,
                base_hw,
                base_hw
            )
        )

        nn.init.trunc_normal_(
            self.pos2d,
            std=0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.enc = nn.TransformerEncoder(
            layer,
            num_layers=depth
        )

        self.out = nn.Conv2d(
            embed_dim,
            in_ch,
            1
        )

    def forward(self, x):

        z = self.proj(x)

        B, C, H, W = z.shape

        pos = F.interpolate(
            self.pos2d,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        z = (
            z + pos
        ).flatten(2).transpose(1, 2)

        z = self.enc(z)

        z = z.transpose(
            1,
            2
        ).reshape(
            B,
            C,
            H,
            W
        )

        return self.out(z)


class TransUNetV2(nn.Module):

    def __init__(self):
        super().__init__()

        self.e1 = DoubleConv(3, 32)
        self.e2 = DoubleConv(32, 64)
        self.e3 = DoubleConv(64, 128)
        self.e4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bridge = DoubleConv(
            256,
            512
        )

        self.trans = TransformerBottleneck(
            512,
            384,
            4,
            6,
            16
        )

        self.u4 = nn.ConvTranspose2d(
            512,
            256,
            2,
            2
        )
        self.d4 = DoubleConv(
            512,
            256
        )

        self.u3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            2
        )
        self.d3 = DoubleConv(
            256,
            128
        )

        self.u2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            2
        )
        self.d2 = DoubleConv(
            128,
            64
        )

        self.u1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            2
        )
        self.d1 = DoubleConv(
            64,
            32
        )

        self.out = nn.Conv2d(
            32,
            1,
            1
        )

        self.aux2 = nn.Conv2d(
            64,
            1,
            1
        )

        self.aux3 = nn.Conv2d(
            128,
            1,
            1
        )

    def forward(self, x):

        e1 = self.e1(x)
        e2 = self.e2(
            self.pool(e1)
        )
        e3 = self.e3(
            self.pool(e2)
        )
        e4 = self.e4(
            self.pool(e3)
        )

        b = self.bridge(
            self.pool(e4)
        )

        b = self.trans(b)

        d4 = self.d4(
            torch.cat(
                [
                    self.u4(b),
                    e4
                ],
                dim=1
            )
        )

        d3 = self.d3(
            torch.cat(
                [
                    self.u3(d4),
                    e3
                ],
                dim=1
            )
        )

        d2 = self.d2(
            torch.cat(
                [
                    self.u2(d3),
                    e2
                ],
                dim=1
            )
        )

        d1 = self.d1(
            torch.cat(
                [
                    self.u1(d2),
                    e1
                ],
                dim=1
            )
        )

        main = self.out(d1)

        if self.training:

            a2 = F.interpolate(
                self.aux2(d2),
                size=main.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            a3 = F.interpolate(
                self.aux3(d3),
                size=main.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            return main, a2, a3

        return main
