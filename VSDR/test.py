import sys
sys.path.append(r"/home/lthpc/Jianxin/SR_Models")
from VDSR.vdsr import Net
import os
from Lib.model_methods import test_model, load_model
from Lib.tool import parse_config_test
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Data_set():
    def __init__(self, numpy_test_path, test_path, output_path, output_numpy_path, ckpt, output_size_w, output_size_j):
        self.numpy_test_path, self.test_path, self.output_path, self.ckpt = numpy_test_path, test_path, output_path, ckpt
        self.output_size_w, self.output_size_j = output_size_w, output_size_j
        self.output_numpy_path = output_numpy_path


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
        learning_rate = float(parse_config_test(config_path, "common_option", "learning_rate"))
        test_path = "/home/lthpc/Jianxin/SR_Models/test_data/4/2016-3/"
        output_path = "/home/lthpc/Jianxin/SR_Models/VDSR/output_image_10/"
        output_path = os.path.join(output_path, area)

        input_size_w = int(parse_config_test(config_path, area, "input_size_w"))
        input_size_j = int(parse_config_test(config_path, area, "input_size_j"))
        output_size_w = int(parse_config_test(config_path, area, "output_size_w"))
        output_size_j = int(parse_config_test(config_path, area, "output_size_j"))
    except Exception as e:
        print("获取配置信息时出错：", e)

    scale = output_size_w // input_size_w + 1
    checkpoint = "/home/lthpc/Jianxin/SR_Models/VDSR/weights-10/EC/4/10km/VDSR_epoch_52.pth"
    print("checkpoint", checkpoint)

    numpy_test_path = "/home/lthpc/Jianxin/SR_Models/numpy_data/test_data/"
    numpy_test_path = os.path.join(numpy_test_path, area)
    numpy_test_path = os.path.join(numpy_test_path, str(scale) + "km")
    test_paths = "/home/lthpc/Jianxin/SR_Models/test_data/4/2016-3/"
    output_path = "/home/lthpc/Jianxin/SR_Models/VDSR/output_image_10/"
    output_path = os.path.join(output_path, area)
    output_path = os.path.join(output_path, str(scale) + "km")
    output_numpy_path = "/home/lthpc/Jianxin/SR_Models/VDSR/output_numpy_10/"
    output_numpy_path = os.path.join(output_numpy_path, area)
    output_numpy_path = os.path.join(output_numpy_path, str(scale) + "km")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("output_path", output_path)

    if not os.path.exists(output_numpy_path):
        os.makedirs(output_numpy_path)
    print("output_numpy_path", output_numpy_path)

    model = Net(num_channels=3, base_filter=16, num_residuals=10, upscale_factor=scale, output_size_w=output_size_w, output_size_j=output_size_j)
    args = Data_set(numpy_test_path, test_path, output_path, output_numpy_path, checkpoint, output_size_w, output_size_j)

    args.model = model
    args.area = area

    # 测试模型
    test_model(args)