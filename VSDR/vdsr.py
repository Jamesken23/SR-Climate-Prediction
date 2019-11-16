import torch
import torch.nn as nn
from math import sqrt
import Lib.tool as tool


class Conv_ReLU_Block(nn.Module):
    def __init__(self, in_filter, out_filter):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_filter, out_channels=out_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(nn.Module):
    def __init__(self, num_channels=3, base_filter=16, num_residuals=10, upscale_factor=5, output_size_w=1, output_size_j=1):
        super(Net, self).__init__()
        self.output_size_w, self.output_size_j = output_size_w, output_size_j
        self.residual_layer = self.make_layer(Conv_ReLU_Block(base_filter, base_filter), num_residuals)
        self.input = nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=base_filter, out_channels=upscale_factor**2, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.res = nn.Conv2d(in_channels=num_channels, out_channels=upscale_factor**2, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # 上采样的倍数
        self.scale = upscale_factor
        self.upscale = nn.PixelShuffle(upscale_factor)

        # 优化器为Adam函数
        self.optimizer = torch.optim.Adam(self.parameters())
        # 定义MSE损失函数
        self.loss_func = torch.nn.MSELoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        residual = self.res(residual)
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        output_image = self.upscale(out)
        # 返回裁剪后的固定尺寸的图片
        return output_image[:, :, :self.output_size_w, :self.output_size_j]