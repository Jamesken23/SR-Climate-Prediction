import torch.nn as nn
import torch
import Lib.tool as tool


class Net(nn.Module):
    def __init__(self, num_channels=3, base_filter=16, upscale_factor=5, output_size_w=1, output_size_j=1):
        super(Net, self).__init__()
        self.output_size_w, self.output_size_j = output_size_w, output_size_j
        self.conv1 = nn.Conv2d(num_channels, base_filter, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(base_filter, base_filter, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(base_filter, base_filter//2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(base_filter//2, upscale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.scale = upscale_factor
        # 优化器为Adam函数
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_func = torch.nn.MSELoss()

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        # out = torch.sigmoid(self.pixel_shuffle(self.conv4(x)))
        out = self.pixel_shuffle(self.conv4(x))
        # 返回裁剪后的固定尺寸的图片
        return out[:, :, :self.output_size_w, :self.output_size_j]
