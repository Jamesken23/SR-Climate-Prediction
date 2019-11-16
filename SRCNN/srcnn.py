import torch
from torch import nn


class Net(nn.Module):
    """
    num_channels: 输入图片的第三维度的channels值
    base_filter： 第一次卷积操作的输出维度值，也是整个三次卷积操作的基准维度
    upscale_factor：输入图片的上采样倍数
    """
    def __init__(self, num_channels=3, base_filter=32, upscale_factor=5, input_size_w=1, input_size_j=1,
                 output_size_w=1, output_size_j=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, base_filter, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(base_filter, base_filter // 2, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(base_filter // 2, upscale_factor ** 2, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.upscale = nn.PixelShuffle(upscale_factor)
        self.scale = upscale_factor
        # 优化器为Adam函数
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_func = torch.nn.MSELoss()

        self.input_size_w, self.input_size_j = input_size_w, input_size_j,
        self.output_size_w, self.output_size_j = output_size_w, output_size_j

    # 构建网络
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        out = self.upscale(x)
        # 返回裁剪后的固定尺寸的图片
        return out[:, :, :self.output_size_w, :self.output_size_j]

