import sys
sys.path.append(r"/home/lthpc/Jianxin/SR_Models")
from LapSRN.lapsrn_simple import Net
import torch
from Lib.model_methods import train_model, train_test
from Lib.tool import parse_config_test
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


class Data_set():
    def __init__(self, Epoch, lr, batch_size, train_path, val_path, dem_path, ckpt, input_size_w, input_size_j,
                output_size_w, output_size_j):
        self.Epoch, self.lr, self.batch_size = Epoch, lr, batch_size
        self.train_path, self.val_path, self.dem_path, self.ckpt = train_path, val_path, dem_path, ckpt
        self.input_size_w, self.input_size_j = input_size_w, input_size_j
        self.output_size_w, self.output_size_j = output_size_w, output_size_j


if __name__ == "__main__":
    # 测试代码为python train.py mode=xxx, area=xxx
    try:
        mode = sys.argv[1].split("=")[1].upper()
        area = sys.argv[2].split("=")[1]
    except Exception:
        print("train.py接收参数错误，类似：python train.py mode=NMC area=3")
        exit()

    last_dir = os.path.dirname(os.getcwd())

    # 获取配置文件路径
    config_path = os.path.join(last_dir, "config")
    config_path = os.path.join(config_path, area)
    config_path = os.path.join(config_path, mode + ".conf")
    print("配置文件路径为:{0}，是否存在：{1}".format(config_path, os.path.exists(config_path)))
    try:
        Epoch = int(parse_config_test(config_path, "common_option", "Epoch"))
        learning_rate = float(parse_config_test(config_path, "common_option", "learning_rate"))
        batch_size = int(parse_config_test(config_path, "common_option", "batch_size"))
        split = float(parse_config_test(config_path, "common_option", "split"))
        train_path = "/home/lthpc/Jianxin/SR_Models/numpy_data/train_data/"
        train_path = os.path.join(train_path, area)
        val_path = "/home/lthpc/Jianxin/SR_Models/numpy_data/val_data/"
        val_path = os.path.join(val_path, area)
        dem_path = os.path.join(last_dir, "dem")
        dem_path = os.path.join(dem_path, area)

        input_size_w = int(parse_config_test(config_path, area, "input_size_w"))
        input_size_j = int(parse_config_test(config_path, area, "input_size_j"))
        mid_size_w = int(parse_config_test(config_path, area, "mid_size_w"))
        mid_size_j = int(parse_config_test(config_path, area, "mid_size_j"))
        output_size_w = int(parse_config_test(config_path, area, "output_size_w"))
        output_size_j = int(parse_config_test(config_path, area, "output_size_j"))
    except Exception as e:
        print("获取配置信息时出错：", e)

    scale = output_size_w // input_size_w + 1
    train_path = os.path.join(train_path, str(scale) + "km")
    val_path = os.path.join(val_path, str(scale) + "km")
    checkpoint = os.path.join(os.getcwd(), "weights-one")
    checkpoint = os.path.join(checkpoint, mode)
    checkpoint = os.path.join(checkpoint, area)
    checkpoint = os.path.join(checkpoint, str(scale) + "km")

    print("checkpoint", checkpoint)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    model = Net(num_channels=3, base_filter=16, num_residuals=10, input_size_w=input_size_w, input_size_j=input_size_j,
                output_size_w=output_size_w, output_size_j=output_size_j)
    # 定义Charbonnier损失函数
    model.loss_func = torch.nn.L1Loss()
    batch_size = 4
    args = Data_set(Epoch, learning_rate, batch_size, train_path, val_path, dem_path, checkpoint, input_size_w, input_size_j,
                    output_size_w, output_size_j)
    args.model = model
    args.model_name = "LapSRN"
    # print("train_data_dir", train_data_dir)
    # print("dem_path", dem_path)
    # 训练模型
    train_model(args)
    # train_test(args)