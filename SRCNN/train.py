import sys
sys.path.append(r"/home/lthpc/Jianxin/SR_Models")
from SRCNN.srcnn import Net
import os
from Lib.save_data import save_numpy
from Lib.model_methods import train_test, train_model
from Lib.tool import parse_config_test
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Data_set():
    def __init__(self, Epoch, lr, batch_size, train_path, val_path, dem_path, ckpt, input_size_w, input_size_j,
                 output_size_w, output_size_j, split, is_training):
        self.Epoch, self.lr, self.batch_size = Epoch, lr, batch_size
        self.train_path, self.val_path, self.dem_path, self.ckpt = train_path, val_path, dem_path, ckpt
        self.input_size_w, self.input_size_j = input_size_w, input_size_j
        self.output_size_w, self.output_size_j = output_size_w, output_size_j
        self.split = split
        self.is_training = is_training


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
        # train_path = "/home/lthpc/Jianxin/SR_Models/train_data/4/2016-5/"
        val_path = "/home/lthpc/Jianxin/SR_Models/numpy_data/val_data/"
        val_path = os.path.join(val_path, area)
        # val_path = "/home/lthpc/Jianxin/SR_Models/validation_data/4/2016-3/"
        dem_path = os.path.join(last_dir, "dem")
        dem_path = os.path.join(dem_path, area)

        input_size_w = int(parse_config_test(config_path, area, "input_size_w"))
        input_size_j = int(parse_config_test(config_path, area, "input_size_j"))
        output_size_w = int(parse_config_test(config_path, area, "output_size_w"))
        output_size_j = int(parse_config_test(config_path, area, "output_size_j"))
    except Exception as e:
        print("获取配置信息时出错：", e)

    scale = output_size_w // input_size_w + 1
    train_path = os.path.join(train_path, str(scale) + "km")
    val_path = os.path.join(val_path, str(scale) + "km")
    checkpoint = os.path.join(os.getcwd(), "weights")
    checkpoint = os.path.join(checkpoint, mode)
    checkpoint = os.path.join(checkpoint, area)
    checkpoint = os.path.join(checkpoint, str(scale) + "km")

    print("checkpoint", checkpoint)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    model = Net(num_channels=3, base_filter=16, upscale_factor=scale, input_size_w=input_size_w, input_size_j=input_size_j,
                 output_size_w=output_size_w, output_size_j=output_size_j)
    Epoch = 100
    args = Data_set(Epoch, learning_rate, batch_size, train_path, val_path, dem_path, checkpoint, input_size_w, input_size_j,
                    output_size_w, output_size_j, split, True)

    args.model = model
    args.model_name = "SRCNN"
    # train_base_dir = "/home/lthpc/Jianxin/SR_Models/numpy_data/train_data"
    # val_base_dir = "/home/lthpc/Jianxin/SR_Models/numpy_data/val_data/"
    # test_base_dir = "/home/lthpc/Jianxin/SR_Models/numpy_data/test_data/"
    # train_base_dir = os.path.join(train_base_dir, area)
    # args.train_base_dir = os.path.join(train_base_dir, str(scale) + "km")
    # val_base_dir = os.path.join(val_base_dir, area)
    # args.val_base_dir = os.path.join(val_base_dir, str(scale) + "km")
    # test_base_dir = os.path.join(test_base_dir, area)
    # args.test_base_dir = os.path.join(test_base_dir, str(scale) + "km")
    # args.test_path = "/home/lthpc/Jianxin/SR_Models/test_data/4/2016-3/"
    # print("train_data_dir", train_data_dir)
    # print("dem_path", dem_path)

    # 训练模型
    # train_test(args)
    train_model(args)
    # save_numpy(args)