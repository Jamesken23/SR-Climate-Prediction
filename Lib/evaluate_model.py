import sys
sys.path.append(r"/home/lthpc/Jianxin/SR_Models")
import copy
import numpy as np
import netCDF4 as nc
from Lib.read_nc import get_data_path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 输出结果
def printResult(predict_results, label_test):
    print("准确率", accuracy_score(predict_results, label_test))
    conf_mat = confusion_matrix(predict_results, label_test)
    # print(conf_mat)
    # print(conf_mat.shape)
    # print(classification_report(predict_results, label_test))
    return conf_mat[-5:, -5:]


# calculate PC PO FAR TS
def calcuTSFARPOS(mat):
    tsList = []
    for i in range(len(mat)):
        ts = mat[i, i] / (np.sum(mat[:, i]) + np.sum(mat[i, :]) - mat[i, i])
        # ts=mat[i,i]/(np.sum(mat[:,i]))
        tsList.append(ts)
    # print("TS", tsList)
    matIsRain = addRainMatrix(mat)
    pc = (matIsRain[0, 0] + matIsRain[1, 1]) / np.sum(matIsRain)
    po = matIsRain[0, 1] / (matIsRain[0, 1] + matIsRain[0, 0])
    far = matIsRain[1, 0] / (matIsRain[1, 0] + matIsRain[0, 0])
    # print("pc:", pc)
    # print("po:", po)
    # print("far:", far)
    return tsList, pc, po, far


# in oder to calculate PC PO FAR ,add up all rain case
def addRainMatrix(mat):
    noRainRow = mat[0, :]
    RainRow = mat[1:, :]
    RainRow = np.sum(RainRow, axis=0)
    matIsRain = np.asarray([[np.sum(RainRow[1:]), np.sum(noRainRow[1:])], [RainRow[0], noRainRow[0]]])
    return matIsRain


# classify 3h
def classify3h(pre0):
    pre = copy.deepcopy(pre0)
    pre[pre < 0.1] = 0
    pre[np.logical_and(pre >= 0.1, pre <= 3)] = 1
    pre[np.logical_and(pre > 3, pre <= 10)] = 2
    pre[np.logical_and(pre > 10, pre <= 20)] = 3
    pre[np.logical_and(pre > 20, pre <= 9990)] = 4
    pre[pre > 9990] = -1
    pre[np.isnan(pre)] = -1
    return pre


# classify 1h
def classify1h(pre0):
    pre = copy.deepcopy(pre0)
    pre[np.isnan(pre)] = -1
    pre[pre < 0.1] = 0
    pre[np.logical_and(pre >= 0.1, pre <= 2.5)] = 1
    pre[np.logical_and(pre > 2.5, pre <= 8)] = 2
    pre[np.logical_and(pre > 8, pre <= 16)] = 3
    pre[np.logical_and(pre > 16, pre <= 9990)] = 4
    pre[pre > 9990] = -1
    return pre


def eval(prediction, label):
    print("pre_data shape", prediction.shape)
    print("label shape", label.shape)
    TS = []
    PC, PO, FAR = 0., 0., 0.
    counter = 0
    for index in range(len(label)):
        # 分别获得等级降水
        pre = classify1h(prediction[index].squeeze())
        lab = classify1h(label[index].squeeze())
        pre_data = pre.reshape([1, prediction.shape[-2]*prediction.shape[-1]]).squeeze()
        lab_data = lab.reshape([1, label.shape[-2]*label.shape[-1]]).squeeze()
        pre_data = pre_data.tolist()
        lab_data = lab_data.tolist()
        # 加入0-5的值，使得能生成5阶矩阵
        pre_data.extend(list(range(5)))
        lab_data.extend(list(range(5)))
        # 获得混淆矩阵
        mat = printResult(pre_data, lab_data)
        tsList, pc, po, far = calcuTSFARPOS(mat)
        TS.append(tsList)
        PC += pc
        PO += po
        FAR += far
        counter += 1
    # 按列求平均值
    print("TS:", np.array(TS).mean(axis=0))
    print("PC:%f, PO:%f, FAR:%f" % (PC / counter, PO / counter, FAR / counter))


def evaluate(pre_dir, label_dir):
    """
    :param pre_dir: 测试得到的数据集文件夹
    :param label_dir: 原始正确标签集文件夹
    :return:
    """
    pre_paths = get_data_path(pre_dir)
    # sorted(pre_paths)
    label_paths = get_data_path(label_dir)
    # sorted(label_paths)
    print("len pre_paths", pre_paths[:5])
    print("len label_paths",label_paths[:5])
    TS = []
    PC, PO, FAR = 0., 0., 0.
    for index in range(len(pre_paths)):
        pre_data = nc.Dataset(pre_paths[index]).variables["precipitation"][:].squeeze()
        label_data = nc.Dataset(label_paths[index]).variables["Total_precipitation_surface"][:].squeeze()
        pre_data = classify1h(pre_data)
        label_data = classify1h(label_data)
        pre = np.reshape(pre_data, newshape=[1, 1401 * 1601]).squeeze()
        label = np.reshape(label_data, newshape=[1, 1401 * 1601]).squeeze()
        pre = pre.tolist()
        label = label.tolist()
        pre.extend(list(range(5)))
        label.extend(list(range(5)))
        mat = printResult(pre, label)
        tsList, pc, po, far = calcuTSFARPOS(mat)
        TS.append(tsList)
        PC += pc
        PO += po
        FAR += far
    # 按列求平均值
    print("TS:", np.array(TS).mean(axis=0))
    print("PC:%f, PO:%f, FAR:%f" % (PC/len(pre_paths), PO/len(pre_paths), FAR/len(pre_paths)))


if __name__ == "__main__":
    pre_dir = "/home/lthpc/Jianxin/SR_Models/ResLap/output_numpy_10/4/2km/prediction.npy"
    label_dir = "/home/lthpc/Jianxin/SR_Models/numpy_data/test_data/4/2km/test_label.npy"
    print("pre_dir", pre_dir)
    print("label_dir", label_dir)
    prediction = np.load(pre_dir)
    label = np.load(label_dir)
    eval(prediction, label)