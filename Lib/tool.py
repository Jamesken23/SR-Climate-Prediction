import configparser
import numpy as np
import scipy.ndimage
import os
from Lib.read_nc import read_data


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
    image = np.zeros(shape=[datas[0].shape[0], datas[0].shape[1], datas[0].shape[2], len(datas)])
    for index in range(len(datas)):
        image[:, :, :, index] = datas[index]
    # 将(batch, height, width, channels)->(batch, channels, width, height)
    new_image = image.transpose(0, 3, 1, 2)
    return new_image


def get_numpy_data(path, mode_1, mode_2):
    if mode_2 == "mid":
        data_file = os.path.join(path, mode_1+"_data"+".npy")
        mid_data_file = os.path.join(path, mode_2+"_"+mode_1+"_label"+".npy")
        label_file = os.path.join(path, mode_1+"_label"+".npy")
        data = np.load(data_file)
        mid_data = np.load(mid_data_file)
        label = np.load(label_file)
        return data, mid_data, label
    else:
        data_file = os.path.join(path, mode_1 + "_data" + ".npy")
        label_file = os.path.join(path, mode_1 + "_label" + ".npy")
        data = np.load(data_file)
        label = np.load(label_file)
        return data, label



# 获取输入数据和标签数据
# 如果是要训练以及验证，返回输入数据和标签；
# 如果是要测试，只返回输入数据
def get_model_data(args, path):
    rain_data, humid_data = read_data(path, key="precipitation")
    dem_data = read_data(args.dem_path, key="dem")
    # print("rain_data", rain_data.shape)
    # print("humid_data", humid_data.shape)
    # print("dem_data", dem_data.shape)
    # 将rain_data中Nan不规范值变为0
    rain_data = not_nan(rain_data)
    # 只保留0-9999mm降水量的数据
    rain_data = np.where(rain_data < 0, 0, rain_data)
    rain_data = np.where(rain_data > 9999, 9999, rain_data)
    # 将humid_data中Nan不规范值变为0
    humid_data = not_nan(humid_data)
    humid_data = normalize_data(humid_data)
    # 将dem_data中Nan不规范值变为0
    dem_data = not_nan(dem_data)
    # 将地形数据归一化
    dem_data = normalize_data(dem_data)

    # 将3D降水数据下采样至(time, input_size_w, input_size_j)
    rain_scale_w = args.input_size_w / rain_data.shape[-2]
    rain_scale_j = args.input_size_j / rain_data.shape[-1]
    LR_rain_data = scipy.ndimage.zoom(rain_data, (1, rain_scale_w, rain_scale_j))

    # 将3D湿度数据下采样至(input_size_w, input_size_j)
    humid_scale_w = args.input_size_w / humid_data.shape[-2]
    humid_scale_j = args.input_size_j / humid_data.shape[-1]
    LR_humid_data = scipy.ndimage.zoom(humid_data, (1, humid_scale_w, humid_scale_j))

    # 将2D地形数据下采样至(input_size_w, input_size_j)
    dem_scale_w = args.input_size_w / dem_data.shape[-2]
    dem_scale_j = args.input_size_j / dem_data.shape[-1]
    LR_dem_data = scipy.ndimage.zoom(dem_data, (1, dem_scale_w, dem_scale_j))
    # 获取融合图片
    input_images = merge_image(LR_rain_data, LR_humid_data, LR_dem_data)
    print("input_images shape", input_images.shape)

    if args.is_training is True:
        rain_shape = rain_data.shape
        rain_datas = rain_data.reshape([rain_shape[0], 1, rain_shape[1], rain_shape[2]])
        return input_images, rain_datas
    else:
        return input_images


# 裁剪图片大小
def crop_image(data, height, width):
    data_shape = data.detach().numpy().shape
    return scipy.ndimage.zoom(data.detach().numpy(), (1, 1, height/data_shape[-2], width/data_shape[-1]))


def outCrop(data, scale):
    if len(data.size()) == 2:
        if scale == 5:
            return data[2:-2,2:-2]
        elif scale == 4:
            return data[1:-2,1:-2]
        elif scale == 3 or scale == 2:
            return data[:-1,:-1]
        else:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ERROR: INVALID UPSCALE FACTOR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    else:
        if scale == 5:
            return data[:,:,2:-2,2:-2]
        elif scale == 4:
            return data[:,:,1:-2,1:-2]
        elif scale == 3 or scale == 2:
            return data[:,:,:-1,:-1]
        else:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ERROR: INVALID UPSCALE FACTOR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


# 根据.conf文件的位置以及当前区域号来确定当前区域的模式名
def parse_config_test(config_path, section, option):
    cp = configparser.ConfigParser()

    if not os.path.exists(config_path):
        print("NO configuration:{0} !".format(config_path))
        exit()
    cp.read(config_path)
    return cp.get(section, option)
