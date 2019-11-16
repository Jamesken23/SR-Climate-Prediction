import netCDF4 as nc
import os
import numpy as np


def get_data_path(path):
    if not os.path.exists(path):
        print(" No such file or directory:{0}".format(path))
        exit()
    # 保存所有子文件名
    all_path = []
    # 开始遍历子文件夹
    for file in os.listdir(path):
        file = os.path.join(path, file)
        all_path.append(file)
    return all_path


def read_data(path="", key=None):
    """
    :param path: 读取文件的位置
    :param key: 文件的类型，降水，地形等.暂定两种类型precipitation 和 dem
    :return:
    """
    rain_data, humid_data, dem_data = [], [], []
    # 首先读取path文件夹下面对应的所有文件名
    if key == "precipitation":
        all_paths = get_data_path(path)
        for path in all_paths:
            datas = nc.Dataset(path)
            rain = datas.variables['Total_precipitation_surface'][:].squeeze()
            humid = datas.variables['Relative_humidity_height_above_ground'][:].squeeze()
            rain_data.append(rain)
            humid_data.append(humid)
            datas.close()
        return np.array(rain_data), np.array(humid_data)
    elif key == "dem":
        all_paths = get_data_path(path)
        for path in all_paths:
            datas = nc.Dataset(path)
            dem = datas.variables['dem'][:].squeeze()
            dem_data.append(dem)
            datas.close()
        return np.array(dem_data)


def get_time_dimen(all_path):
    paths = get_data_path(all_path)
    if len(paths) == 0:
        print("There is no file in the:{0}".format(all_path))
        exit()
    dataset = nc.Dataset(paths[0])

    time = dataset.variables["time"][:]
    # *60.0 means change hours into minutes
    time = time * 60.0
    dataset.close()
    return time


def get_lat_lon_dimen(all_path):
    paths = get_data_path(all_path)
    if len(paths) == 0:
        print("There is no file in the:{0}".format(all_path))
        exit()
    dataset = nc.Dataset(paths[0])
    lat = dataset.variables["lat"][:]
    lon = dataset.variables["lon"][:]
    dataset.close()
    return lat, lon


def get_name(all_path, area):
    paths = get_data_path(all_path)
    # 文件名排序
    # paths = sorted(paths)
    if len(paths) == 0:
        print("There is no file in the:{0}".format(all_path))
        exit()
    dataset = nc.Dataset(paths[0])
    data = dataset.variables["time"][:]
    # predit_hour = str(data[-1])
    names = [path.split('/')[-1] for path in paths]
    names_new = []
    for name in names:
        l = name.split("_")
        mode_str = l[0]
        # time_str: xxx.nc
        time_str = l[1]
        # new_name = "MSP2_PMSC_AIWSRPF_" + mode_str + "SP1_L88_" + area_str + "_" + time_str + ".nc".format(
        #     predit_hour)
        new_name = "%s_%s_%s" % (mode_str, area, time_str)
        names_new.append(new_name)
    return names_new