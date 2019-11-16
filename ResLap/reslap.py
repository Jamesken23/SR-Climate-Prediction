import torch
import math
import numpy as np
import torch.nn as nn
import Lib.tool as tool
from torch.autograd import Variable


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class RDB_Conv(nn.Module):
    def __init__(self, in_filter, out_filter):
        super(RDB_Conv, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_filter, out_filter, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        # return torch.cat([x, out], 1)
        return torch.add(x, out)


class RDB(nn.Module):
    def __init__(self, in_filter, out_filter, n_block):
        super(RDB, self).__init__()
        convs = []
        for c in range(n_block):
            convs.append(RDB_Conv(in_filter, out_filter))
        self.convs = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(in_filter, out_filter, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.LFF(self.convs(x))


class Net(nn.Module):
    def __init__(self, num_channels=3, base_filter=16, num_residuals=10, input_size_w=1, input_size_j=1,
                 mid_size_w=1, mid_size_j=1, output_size_w=1, output_size_j=1):
        super(Net, self).__init__()
        self.input_size_w, self.input_size_j = input_size_w, input_size_j
        self.mid_size_w, self.mid_size_j = mid_size_w, mid_size_j
        self.output_size_w, self.output_size_j = output_size_w, output_size_j
        self.first_scale = self.mid_size_w // self.input_size_w + 1
        self.second_scale = self.output_size_w // self.mid_size_w + 1
        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # 对输入图片做反卷积
        self.convt_I1 = nn.ConvTranspose2d(in_channels=num_channels, out_channels=1, kernel_size=4, stride=self.first_scale, padding=1, bias=False)
        self.convt_R1 = nn.Conv2d(in_channels=base_filter, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = RDB(base_filter, base_filter, num_residuals)
        # 第二次对输入图片做反卷积
        self.convt_I2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=self.second_scale, padding=1,
                                           bias=False)
        self.convt_R2 = nn.Conv2d(in_channels=base_filter, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = RDB(base_filter, base_filter, num_residuals)

        self.first_deconv = nn.ConvTranspose2d(in_channels=base_filter, out_channels=base_filter, kernel_size=4,
                                            stride=self.first_scale, padding=1, bias=False)
        self.second_deconv = nn.ConvTranspose2d(in_channels=base_filter, out_channels=base_filter, kernel_size=4,
                                             stride=self.second_scale, padding=1, bias=False)
        # 优化器为Adam函数
        self.optimizer = torch.optim.Adam(self.parameters())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        input = self.relu(self.conv_input(x))
        convt_F1 = self.convt_F1(input)
        convt_F1 = self.relu(self.first_deconv(convt_F1))
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1
        HR_2x = HR_2x[:, :, :self.mid_size_w, :self.mid_size_j]
        # HR_2x_numpy = tool.crop_image(HR_2x, self.mid_size_w, self.mid_size_j)
        # HR_2x = torch.from_numpy(HR_2x_numpy).float()
        # HR_2x = Variable(HR_2x)

        # convt_F1_numpy = tool.crop_image(convt_F1, self.mid_size_w, self.mid_size_j)
        # convt_F1 = torch.from_numpy(convt_F1_numpy).float()
        # convt_F1 = Variable(convt_F1)
        convt_F1 = convt_F1[:, :, :self.mid_size_w, :self.mid_size_j]

        convt_F2 = self.convt_F2(convt_F1)
        convt_F2 = self.relu(self.second_deconv(convt_F2))
        convt_I2 = self.convt_I2(HR_2x)
        convt_R2 = self.convt_R2(convt_F2)
        HR_4x = convt_I2 + convt_R2
        # HR_4x_numpy = tool.crop_image(HR_4x, self.output_size_w, self.output_size_j)
        # HR_4x = torch.from_numpy(HR_4x_numpy).float()
        # HR_4x = Variable(HR_4x)
        HR_4x = HR_4x[:, :, :self.output_size_w, :self.output_size_j]
        return HR_2x, HR_4x


class L2_Charbonnier_loss(nn.Module):
    """L2 Charbonnierloss."""

    def __init__(self):
        super(L2_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = diff * diff + self.eps
        loss = torch.sum(error)
        return loss