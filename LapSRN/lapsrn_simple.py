import torch
import math
import numpy as np
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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


class Conv_ReLU_Block(nn.Module):
    def __init__(self, in_filter, out_filter):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_filter, out_channels=out_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(nn.Module):
    def __init__(self, num_channels=3, base_filter=16, num_residuals=10, input_size_w=1, input_size_j=1,
                output_size_w=1, output_size_j=1):
        super(Net, self).__init__()
        self.input_size_w, self.input_size_j = input_size_w, input_size_j
        self.output_size_w, self.output_size_j = output_size_w, output_size_j
        self.upscale = self.output_size_w // self.input_size_w + 1
        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # 对输入图片做反卷积
        self.convt_I1 = nn.ConvTranspose2d(in_channels=num_channels, out_channels=1, kernel_size=4, stride=self.upscale,
                                           padding=1,bias=False)
        self.convt_R1 = nn.Conv2d(in_channels=base_filter, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(Conv_ReLU_Block(base_filter, base_filter), num_residuals)

        self.first_deconv = nn.ConvTranspose2d(in_channels=base_filter, out_channels=base_filter, kernel_size=4,
                                                stride=self.upscale, padding=1, bias=False)

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

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        input = self.relu(self.conv_input(x))

        convt_F1 = self.convt_F1(input)
        convt_F1 = self.relu(self.first_deconv(convt_F1))
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1

        return HR_2x[:, :, :self.output_size_w, :self.output_size_j]