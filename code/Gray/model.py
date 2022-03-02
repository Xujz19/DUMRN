import torch
import torch.nn as nn
import torch.nn.functional as F
from antialias import Downsample


def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels,   1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                Downsample(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(Downsample(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


##########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
##---------- Dual Attention Unit (DAU) ----------
class DAU(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(DAU, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, bias=True):
        super(ResidualLayer, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True),
        )

        self.body = DAU(in_channels)

        self.end = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        out = self.end(res + x)
        return out


class DenseFusion(nn.Module):
    def __init__(self, in_channels):
        super(DenseFusion, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.PReLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels  * 2, in_channels  * 4, kernel_size=3, stride=2, padding=1, bias=True),
            nn.PReLU(),
        )


        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.PReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.PReLU(),
        )


    def forward(self, x1, x2, x3):
        ft_fusion = x1

        ft = self.down1(ft_fusion)
        ft = self.down2(ft)
        ft = ft - x3
        ft = self.up2(ft)
        ft = self.up1(ft)
        ft_fusion = ft_fusion + ft

        ft = self.down1(ft_fusion)
        ft = ft - x2
        ft = self.up1(ft)
        ft_fusion = ft_fusion + ft

        return ft_fusion


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleBlock, self).__init__()

        self.down1 = ResidualDownSample(in_channels)
        self.down2 = ResidualDownSample(in_channels * 2)

        self.layer1 = ResidualLayer(in_channels)
        self.layer2 = ResidualLayer(in_channels * 2)
        self.layer3 = ResidualLayer(in_channels * 4)

        self.fusion = DenseFusion(in_channels)

    def forward(self, x):
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        x3 = self.layer3(x3)

        out = self.fusion(x1, x2, x3)
        out = out + x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, n_feats, residual=True, bias=True):
        super(ResidualBlock, self).__init__()

        self.residual = residual

        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=bias),
        )

    def forward(self, x):
        res = self.body(x)
        if self.residual:
            res += x
        return res


class MSFN(nn.Module):
    def __init__(self, n_colors, in_channels, n_blocks, residual=False):
        super(MSFN, self).__init__()

        self.residual = residual
        self.head = nn.Conv2d(n_colors, in_channels, kernel_size=3, stride=1, padding=1)

        self.regularizer1 = MultiScaleBlock(in_channels)
        self.alpha1 = ResidualBlock(in_channels, residual=False)

        self.regularizer2 = MultiScaleBlock(in_channels)
        self.alpha2 = ResidualBlock(in_channels, residual=False)
        self.beta2 = ResidualBlock(in_channels, residual=False)

        self.regularizer3 = MultiScaleBlock(in_channels)
        self.alpha3 = ResidualBlock(in_channels, residual=False)
        self.beta3 = ResidualBlock(in_channels, residual=False)

        self.regularizer4 = MultiScaleBlock(in_channels)
        self.alpha4 = ResidualBlock(in_channels, residual=False)
        self.beta4 = ResidualBlock(in_channels, residual=False)

        self.regularizer5 = MultiScaleBlock(in_channels)
        self.alpha5 = ResidualBlock(in_channels, residual=False)
        self.beta5 = ResidualBlock(in_channels, residual=False)

        self.regularizer6 = MultiScaleBlock(in_channels)
        self.alpha6 = ResidualBlock(in_channels, residual=False)
        self.beta6 = ResidualBlock(in_channels, residual=False)

        self.end = nn.Conv2d(in_channels, n_colors, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.head(x)

        reg = self.regularizer1(x0)
        adj = 0
        grad = adj - reg
        x1 = x0 + self.alpha1(grad)

        reg = self.regularizer2(x1)
        adj = x1 - x0
        grad = adj - reg
        x2 = x1 + self.alpha2(grad) + self.beta2(x1 - x0)

        reg = self.regularizer3(x2)
        adj = x2- x0
        grad = (adj - reg)
        x3 = x2 + self.alpha3(grad) + self.beta3(x2 - x1)

        reg = self.regularizer4(x3)
        adj = x3- x0
        grad = (adj - reg)
        x4 = x3 + self.alpha4(grad) + self.beta4(x3 - x2)

        reg = self.regularizer5(x4)
        adj = x4- x0
        grad = (adj - reg)
        x5 = x4 + self.alpha5(grad) + self.beta5(x4 - x3)

        reg = self.regularizer6(x5)
        adj = x5- x0
        grad = (adj - reg)
        x6 = x5 + self.alpha6(grad) + self.beta6(x5 - x4)

        res = self.end(x6)
        if self.residual:
            out = res + x
        else:
            out = res

        return out



