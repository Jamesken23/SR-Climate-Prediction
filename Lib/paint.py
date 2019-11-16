import sys
sys.path.append(r"D:/PyCharm-Community/Workplace/SR_Models")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gp
import netCDF4 as nc


def get_data(path):
    datas = nc.Dataset(path)
    rain = datas.variables['Total_precipitation_surface'][:].squeeze()
    # rain = datas.variables['precipitation'][:].squeeze()
    return rain


def graph_data(path):
    cmap_list = ['#FFFFFF', '#ECF6FF', '#B5CAFF', '#8CB1FF', '#7185FA', '#6370FA', '#3ABB3C', '#B4D26E', '#B9F971',
                 '#E1A20E', '#E40000', '#800000', '#000000']
    # allPhaseArr = np.load(path)[0][0]
    allPhaseArr = get_data(path)
    print(allPhaseArr.shape)

    fig = plt.figure(figsize=[8, 7])
    ax = fig.add_subplot(1, 1, 1)
    china_map = gp.GeoDataFrame.from_file("D:/PyCharm-Community/Workplace/SR_Models/data/provinceLine.shx", encoding='gb18030')

    geo_ploy = china_map['geometry']

    ax.set_aspect('equal')
    # 几何图形绘制
    geo_ploy.plot(ax=ax, color='gray')

    plt.imshow(allPhaseArr, extent=(108, 124, 22, 36), cmap=mpl.colors.ListedColormap(cmap_list))
    x_ticks = ax.set_xticks([108, 110, 112, 114, 116, 118, 120, 122, 124])
    # x_labels = ax.set_xticklabels(["108°E", "111°E", "114°E", "117°E", "120°E", "122°E", "124°E"], fontsize="large")
    x_labels = ax.set_xticklabels(["108°E", "110°E", "112°E", "114°E", "116°E", "108°E", "120°E", "122°E", "124°E"], fontsize="large")
    y_ticks = ax.set_yticks([36, 34, 32, 30, 28, 26, 24, 22])
    # y_labels = ax.set_yticklabels(["36°N", "33°N", "30°N", "27°N", "24°N", "22°N"], fontsize="large")
    y_labels = ax.set_yticklabels(["36°N", "34°N", "32°N", "30°N", "28°N", "26°N", "24°N", "22°N"], fontsize="large")

    # plt.imshow(allPhaseArr, extent=(104, 124, 15, 27), cmap=mpl.colors.ListedColormap(cmap_list))
    # x_ticks = ax.set_xticks([104, 108, 112, 116, 120,  124])
    # x_labels = ax.set_xticklabels(["104°E", "108°E", "112°E", "116°E", "120°E", "124°E"], fontsize="large")
    # y_ticks = ax.set_yticks([27, 25, 23, 21, 19, 17, 15])
    # y_labels = ax.set_yticklabels(["27°N", "25°N", "21°N", "19°N", "17°N", "15°N"], fontsize="large")

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
    plt.savefig("D:/PyCharm-Community/Workplace/SR_Models/data/NCN_2017060505_ResLap.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # path_1 = "E:/numpy_data/2km/test_data.npy"
    path_1 = "D:/PyCharm-Community/Workplace/SR_Models/data/NCN_3_2017060505 ResLap.nc"
    graph_data(path_1)