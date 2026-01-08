import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, ic, oc, k=(3, 3), p="same", act=True):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, kernel_size=k, padding=p)
        self.bn = nn.BatchNorm2d(oc)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.bn(self.act(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        self.scale = nn.Parameter(torch.zeros(1))
        
        self._init_weights()

    def _init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        att = self.sigmoid(out)
        return 1.0 + self.scale * (att - 1.0)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.scale = nn.Parameter(torch.zeros(1))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        att = self.sigmoid(out)
        return 1.0 + self.scale * (att - 1.0)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class TrackNetAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d_1 = Conv(9, 64)
        self.conv2d_2 = Conv(64, 64)
        self.cbam_1 = CBAM(64)
        self.max_pooling_1 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2d_3 = Conv(64, 128)
        self.conv2d_4 = Conv(128, 128)
        self.cbam_2 = CBAM(128)
        self.max_pooling_2 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2d_5 = Conv(128, 256)
        self.conv2d_6 = Conv(256, 256)
        self.conv2d_7 = Conv(256, 256)
        self.cbam_3 = CBAM(256)
        self.max_pooling_3 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2d_8 = Conv(256, 512)
        self.conv2d_9 = Conv(512, 512)
        self.conv2d_10 = Conv(512, 512)
        self.cbam_4 = CBAM(512)

        self.up_sampling_1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv2d_11 = Conv(768, 256)
        self.conv2d_12 = Conv(256, 256)
        self.conv2d_13 = Conv(256, 256)
        self.cbam_5 = CBAM(256)

        self.up_sampling_2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv2d_14 = Conv(384, 128)
        self.conv2d_15 = Conv(128, 128)
        self.cbam_6 = CBAM(128)

        self.up_sampling_3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv2d_16 = Conv(192, 64)
        self.conv2d_17 = Conv(64, 64)
        self.cbam_7 = CBAM(64)
        self.conv2d_18 = nn.Conv2d(64, 3, kernel_size=(1, 1), padding='same')

    def forward(self, x):
        x = self.conv2d_1(x)
        x1 = self.conv2d_2(x)
        x1 = self.cbam_1(x1)
        x = self.max_pooling_1(x1)

        x = self.conv2d_3(x)
        x2 = self.conv2d_4(x)
        x2 = self.cbam_2(x2)
        x = self.max_pooling_2(x2)

        x = self.conv2d_5(x)
        x = self.conv2d_6(x)
        x3 = self.conv2d_7(x)
        x3 = self.cbam_3(x3)
        x = self.max_pooling_3(x3)

        x = self.conv2d_8(x)
        x = self.conv2d_9(x)
        x = self.conv2d_10(x)
        x = self.cbam_4(x)

        x = self.up_sampling_1(x)
        x = torch.concat([x, x3], dim=1)

        x = self.conv2d_11(x)
        x = self.conv2d_12(x)
        x = self.conv2d_13(x)
        x = self.cbam_5(x)

        x = self.up_sampling_2(x)
        x = torch.concat([x, x2], dim=1)

        x = self.conv2d_14(x)
        x = self.conv2d_15(x)
        x = self.cbam_6(x)

        x = self.up_sampling_3(x)
        x = torch.concat([x, x1], dim=1)

        x = self.conv2d_16(x)
        x = self.conv2d_17(x)
        x = self.cbam_7(x)
        x = self.conv2d_18(x)

        x = torch.sigmoid(x)

        return x
