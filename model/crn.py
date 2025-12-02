import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), stride=(2,2), padding=(1,1)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class CRN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # encoder
        self.enc1 = ConvBlock(in_channels, 16)  # -> C=16
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        # bottleneck conv to reduce channel dim
        self.bottleneck = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1))
        # GRU (we will flatten freq*channel as feature dimension)
        # freq after conv: n_fft/2+1 reduced by stride factors (2,2),(2,2),(2,2) on freq dim approx
        # We'll compute runtime
        # decoder
        self.dec3 = DeconvBlock(64, 32)
        self.dec2 = DeconvBlock(32, 16)
        self.dec1 = DeconvBlock(16, 8)
        self.out_conv = nn.Conv2d(8, 1, kernel_size=(1,1))
        self.sigmoid = nn.Sigmoid()

        self.gru = None  # will init lazily when input size known

    def forward(self, x):
        # x: (B, 1, F, T)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        # b: (B, C, F', T')
        B, C, Fp, Tp = b.shape
        # prepare for GRU: permute to (B, T', C*F')
        b_perm = b.permute(0, 3, 1, 2).contiguous()  # (B, T', C, F')
        feat = b_perm.view(B, Tp, C * Fp)
        # lazy init of GRU
        if self.gru is None:
            self.gru = nn.GRU(input_size=C*Fp, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True).to(x.device)
        g_out, _ = self.gru(feat)  # (B, T', hidden*2)
        # project back to shape
        proj = g_out.view(B, Tp, C, Fp).permute(0, 2, 3, 1).contiguous()  # (B, C, F', T')
        d3 = self.dec3(proj)
        d2 = self.dec2(d3 + e2)
        d1 = self.dec1(d2 + e1)
        out = self.out_conv(d1)
        mask = self.sigmoid(out)
        # output shape (B,1,F,T) - mask in linear domain to multiply with magnitude
        return mask