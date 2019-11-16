import os
import netCDF4 as nc
import numpy as np
import scipy.ndimage
from Lib.read_nc import get_data_path
from torch.utils.data import Dataset


# 将数据归一化
def normalize_data(data):
    """
    :param data: numpy数组
    :return: data的归一化数组
    """
    min_value = data.min()
    max_value = data.max()
    return (data - min_value)/(max_value - min_value)


# 去除数据中的Nan不规范值，转化为0
def not_nan(data):
    nan_index = np.isnan(data)
    data[nan_index] = 0
    return data


# 融合多气候要素，形成模型输入图片
def merge_image(*datas):
    # 定义最后融合后的图片
    image = np.zeros(shape=[datas[0].shape[0], datas[0].shape[1], len(datas)])
    for index in range(len(datas)):
        image[:, :, index] = datas[index]
    # 将(height, width, channels)->(channels, width, height)
    new_image = image.transpose(2, 0, 1)
    return new_image


class LAPSLoader(Dataset):
    def __init__(self, args, data_path):
        self.args = args
        self.data_dir = data_path
        dem_path = get_data_path(args.dem_path)
        self.dem_data = nc.Dataset(dem_path[0]).variables["dem"][:]
        self.fileList = os.listdir(data_path)
        # 文件排序
        self.fileList.sort()

    def __getitem__(self, index):
        """===without augmentation==="""
        filePath = self.fileList[index]
        filePath = os.path.join(self.data_dir, filePath)
        data = nc.Dataset(filePath)
        rain_data = data.variables['Total_precipitation_surface'][:].squeeze()
        humid_data = data.variables['Relative_humidity_height_above_ground'][:].squeeze()

        # 将rain_data中Nan不规范值变为0
        rain_data = not_nan(rain_data)
        # 只保留0-9999mm降水量的数据
        rain_data = np.where(rain_data < 0, 0, rain_data)
        rain_data = np.where(rain_data > 9999, 9999, rain_data)
        # 将humid_data中Nan不规范值变为0
        humid_data = not_nan(humid_data)
        humid_data = normalize_data(humid_data)
        # 将dem_data中Nan不规范值变为0
        dem_data = not_nan(self.dem_data)
        # 将地形数据归一化
        dem_data = normalize_data(dem_data)

        # 将3D降水数据下采样至(time, input_size_w, input_size_j)
        rain_scale_w = self.args.input_size_w / rain_data.shape[-2]
        rain_scale_j = self.args.input_size_j / rain_data.shape[-1]
        LR_rain_data = scipy.ndimage.zoom(rain_data, (rain_scale_w, rain_scale_j))

        # 将3D湿度数据下采样至(input_size_w, input_size_j)
        humid_scale_w = self.args.input_size_w / humid_data.shape[-2]
        humid_scale_j = self.args.input_size_j / humid_data.shape[-1]
        LR_humid_data = scipy.ndimage.zoom(humid_data, (humid_scale_w, humid_scale_j))

        # 将2D地形数据下采样至(input_size_w, input_size_j)
        dem_scale_w = self.args.input_size_w / dem_data.shape[-2]
        dem_scale_j = self.args.input_size_j / dem_data.shape[-1]
        LR_dem_data = scipy.ndimage.zoom(dem_data, (dem_scale_w, dem_scale_j))
        # 获取融合图片
        input_images = merge_image(LR_rain_data, LR_humid_data, LR_dem_data)
        rain_shape = rain_data.shape
        rain_datas = rain_data.reshape([1, rain_shape[0], rain_shape[1]])
        return input_images.astype(dtype=np.float32), rain_datas.astype(dtype=np.float32)

    def __len__(self):
        return len(self.fileList)


# if __name__ == "__main__":
#     last_dir = os.path.dirname(os.getcwd())
#     path = os.path.join(last_dir, "dem")
#     path = os.path.join(path, "3")
#     dataset = LAPSLoaderWo(path)
#     print(dataset)
#     item = dataset.__getitem__(0)
#     print(item.shape)

    # path_1 = r"D:\PyCharm-Community\Workplace\SR_Models\dem\3\dem_3.nc"
    # data = nc.Dataset(path_1)
    # print("data keys", data.variables["dem"][:].shape)