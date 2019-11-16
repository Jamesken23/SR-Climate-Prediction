from math import inf
import torch
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
import scipy.ndimage
import Lib.tool as tool
from Lib.loader import LAPSLoader
import scipy.ndimage
import Lib.write_nc as write_nc
import Lib.read_nc as read_nc
import os


def train_multi_models(args):
    # 获取训练数据集
    train_data, mid_train_label, train_label = tool.get_numpy_data(args.train_path, "train", "mid")
    val_data, mid_val_label, val_label = tool.get_numpy_data(args.val_path, "val", "mid")
    print("train_data shape", train_data.shape)
    print("mid_train_label shape", mid_train_label.shape)
    print("train_label shape", train_label.shape)
    print("val_data shape", val_data.shape)
    print("mid_val_label shape", mid_val_label.shape)
    print("val_label shape", val_label.shape)
    # 先将numpy数组转化为tensor张量
    train_data = torch.from_numpy(train_data).float()
    mid_train_label = torch.from_numpy(mid_train_label).float()
    train_label = torch.from_numpy(train_label).float()
    # 将数据放在数据库中
    torch_dataset = Data.TensorDataset(train_data, mid_train_label, train_label)
    # 将训练集转化为模型可加载的格式, config.batch_size
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=True)
    # 加载模型
    model = args.model
    history_epoch = 0
    # 定义最小损失值
    min_loss = inf
    try:
        checkpoint = torch.load(load_model(args))
        print("加载的模型为：", load_model(args))
        model.load_state_dict(checkpoint['net'])
        history_epoch = checkpoint['epoch']
        min_loss = checkpoint['min_loss']
        print("已加载第%d次Epoch的模型" % history_epoch)
    except:
        print("未加载模型")
    # 使用GPU训练
    if torch.cuda.is_available():
        print("使用GPU训练模型")
        model = model.cuda()
        loss_func = model.loss_func.cuda()
    # 启用 BatchNormalization 和 Dropout
    model.train()
    print("# training start-----------------------------")
    for epoch in range(history_epoch, args.Epoch):
        # 每一轮的平均损失
        total_loss = 0.
        # 计数
        counters = 0
        for batch_x, batch_mid, batch_y in train_loader:
            if torch.cuda.is_available():
                batch_x = Variable(batch_x.cuda())
                batch_mid = Variable(batch_mid.cuda())
                batch_y = Variable(batch_y.cuda())
            counters += 1
            first_output, second_output = model(batch_x)
            first_loss = loss_func(first_output, batch_mid)
            second_loss = loss_func(second_output, batch_y)
            total_loss += float(first_loss) + float(second_loss)
            # 把梯度置零
            model.optimizer.zero_grad()
            first_loss.backward(retain_graph=True)
            second_loss.backward()
            model.optimizer.step()
        print("Epoch:%d, Train Loss:%f" % (epoch, total_loss/counters))
        # 验证双层模型，并输出L1损失值
        val_loss = val_multi_model(model, val_data, mid_val_label, val_label, args.batch_size, loss_func)
        print("Epoch:%d, Validation Loss:%f" % (epoch, val_loss))
        if val_loss < min_loss:
            # 保存模型
            save_model(args, model, epoch, min_loss)
            min_loss = val_loss
    print("# training end-----------------------------")


def train_test(args):
    train_loader = Data.DataLoader(dataset=LAPSLoader(args, args.train_path), batch_size=args.batch_size, shuffle=True)
    val_loader = Data.DataLoader(dataset=LAPSLoader(args, args.val_path), batch_size=args.batch_size, shuffle=True)
    # 加载模型
    model = args.model
    history_epoch = 0
    # 定义最小损失值
    min_loss = inf
    try:
        checkpoint = torch.load(load_model(args))
        print("加载的模型为：", load_model(args))
        model.load_state_dict(checkpoint['net'])
        history_epoch = checkpoint['epoch']
        min_loss = checkpoint['min_loss']
        print("已加载第%d次Epoch的模型" % history_epoch)
    except:
        print("未加载模型")
    # 使用GPU训练
    if torch.cuda.is_available():
        print("使用GPU训练模型")
        model = model.cuda()
    # 启用 BatchNormalization 和 Dropout
    model.train()

    print("# training start-----------------------------")
    for epoch in range(history_epoch, args.Epoch):
        # 每一轮的平均损失
        total_loss = 0.
        # 计数
        counters = 0
        for batch_x, batch_y in train_loader:
            batch_x = Variable(batch_x.cuda())
            batch_y = Variable(batch_y.cuda())
            # if torch.cuda.is_available():
            #     print("使用GPU读取数据")
            #     batch_x = batch_x.cuda()
            #     batch_y = batch_y.cuda()
            counters += 1
            output = model(batch_x)
            mse_loss = model.loss_func(output, batch_y)
            total_loss += mse_loss
            # 把梯度置零
            model.optimizer.zero_grad()
            mse_loss.backward()
            model.optimizer.step()
        print("Epoch:%d, Train Loss:%f" % (epoch, total_loss/counters))
        # 验证模型，并输出MSE损失值
        val_loss = val_model(model, val_loader)
        print("Epoch:%d, Validation Loss:%f" % (epoch, val_loss))
        if val_loss < min_loss:
            # 保存模型
            save_model(args, model, epoch, min_loss)
            min_loss = val_loss
    print("# training end-----------------------------")


def train_model(args):
    # 获取训练数据集
    train_data, train_label = tool.get_numpy_data(args.train_path, "train", "")
    val_data, val_label = tool.get_numpy_data(args.val_path, "val", "")
    print("train_data shape", train_data.shape)
    print("train_label shape", train_label.shape)
    print("val_data shape", val_data.shape)
    print("val_label shape", val_label.shape)
    # 先将numpy数组转化为tensor张量
    train_data = torch.from_numpy(train_data).float()
    train_label = torch.from_numpy(train_label).float()
    # 将数据放在数据库中
    torch_dataset = Data.TensorDataset(train_data, train_label)
    # 将训练集转化为模型可加载的格式, config.batch_size
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=True)
    # 加载模型
    model = args.model
    history_epoch = 0
    # 定义最小损失值
    min_loss = inf
    try:
        checkpoint = torch.load(load_model(args))
        print("加载的模型为：", load_model(args))
        model.load_state_dict(checkpoint['net'])
        history_epoch = checkpoint['epoch']
        min_loss = checkpoint['min_loss']
        print("已加载第%d次Epoch的模型" % history_epoch)
    except:
        print("未加载模型")
    # 使用GPU训练
    if torch.cuda.is_available():
        print("使用GPU训练模型")
        model = model.cuda()
        loss_func = model.loss_func.cuda()
    # 启用 BatchNormalization 和 Dropout
    model.train()
    print("# training start-----------------------------")
    for epoch in range(history_epoch, args.Epoch):
        # 每一轮的平均损失
        total_loss = 0.
        # 计数
        counters = 0
        for batch_x, batch_y in train_loader:
            if torch.cuda.is_available():
                batch_x = Variable(batch_x.cuda())
                batch_y = Variable(batch_y.cuda())
            counters += 1
            output = model(batch_x)
            mse_loss = loss_func(output, batch_y)
            total_loss += float(mse_loss)
            # 把梯度置零
            model.optimizer.zero_grad()
            mse_loss.backward()
            model.optimizer.step()
        print("Epoch:%d, Train Loss:%f" % (epoch, total_loss/counters))
        # 验证模型，并输出MSE损失值
        val_loss = validation(model, val_data, val_label, args.batch_size, loss_func)
        print("Epoch:%d, Validation Loss:%f" % (epoch, val_loss))
        if val_loss < min_loss:
            # 保存模型
            save_model(args, model, epoch, min_loss)
            min_loss = val_loss
    print("# training end-----------------------------")


def val_model(model, val_loader):
    total_loss = 0.
    counter = 0
    for val_data, val_label in val_loader:
        counter += 1
        val_data = Variable(val_data)
        val_label = Variable(val_label)
        output = model(val_data)
        val_loss = model.loss_func(output, val_label)
        total_loss += val_loss
    return total_loss/counter


# 验证模型
def val_multi_model(model, val_data, mid_val_label, val_label, batch_size, loss_func):
    val_data = torch.from_numpy(val_data).float()
    mid_val_label = torch.from_numpy(mid_val_label).float()
    val_label = torch.from_numpy(val_label).float()
    val_dataset = Data.TensorDataset(val_data, mid_val_label, val_label)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0.
    counter = 0
    for batch_x, batch_mid, batch_y in val_loader:
        counter += 1
        batch_x = Variable(batch_x.cuda())
        batch_mid = Variable(batch_mid.cuda())
        batch_y = Variable(batch_y.cuda())
        # 返回输出值
        first_output, second_output = model(batch_x)
        first_loss = loss_func(first_output, batch_mid)
        second_loss = loss_func(second_output, batch_y)
        total_loss += float(first_loss) + float(second_loss)
    return total_loss/counter


# 验证模型
def validation(model, val_data, val_label, batch_size, loss_func):
    val_data = torch.from_numpy(val_data).float()
    val_label = torch.from_numpy(val_label).float()
    val_dataset = Data.TensorDataset(val_data, val_label)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0.
    counter = 0
    for batch_x, batch_y in val_loader:
        counter += 1
        batch_x = Variable(batch_x.cuda())
        batch_y = Variable(batch_y.cuda())
        # 返回输出值
        output = model(batch_x)
        val_loss = loss_func(output, batch_y)
        total_loss += float(val_loss)
    return total_loss/counter


# 测试双层模型
def test_multi_model(args):
    names = read_nc.get_name(args.test_path, args.area)
    all_dimen = []
    time_dimen = read_nc.get_time_dimen(args.test_path)
    lat, lon = read_nc.get_lat_lon_dimen(args.test_path)
    all_dimen.append(time_dimen)
    all_dimen.append(lat)
    all_dimen.append(lon)
    # 获取训练数据集
    test_data, test_label = tool.get_numpy_data(args.numpy_test_path, "test", "")
    print("test_data shape", test_data.shape)
    print("test_label shape", test_label.shape)
    test_data = torch.from_numpy(test_data).float()
    test_label = torch.from_numpy(test_label).float()
    torch_dataset = Data.TensorDataset(test_data, test_label)
    # 将训练集转化为模型可加载的格式, config.batch_size
    test_loader = Data.DataLoader(dataset=torch_dataset, batch_size=1)
    # 加载模型
    model = args.model
    try:
        checkpoint = torch.load(args.ckpt)
        print("加载的模型为：", args.ckpt)
        model.load_state_dict(checkpoint['net'])
        history_epoch = checkpoint['epoch']
        print("已加载第%d次Epoch的模型" % history_epoch)
    except:
        print("未加载模型")
    # 使用GPU训练
    loss_func = torch.nn.MSELoss()
    if torch.cuda.is_available():
        print("使用GPU训练模型")
        model = model.cuda()
        loss_func = loss_func.cuda()
    # 不启用 BatchNormalization 和 Dropout
    model.eval()
    counters = 0
    total_loss = 0.
    output_images = []
    print("testing start-----------------------------")
    for batch_x, batch_y in test_loader:
        if torch.cuda.is_available():
            batch_x = Variable(batch_x.cuda())
            batch_y = Variable(batch_y.cuda())
        counters += 1
        mid_output, output = model(batch_x)
        rmse_loss = torch.sqrt(loss_func(output, batch_y))
        total_loss += float(rmse_loss)

        output = output.cpu().detach().numpy()
        outnc = [i for i in all_dimen]
        outnc.append(output)
        writepath = os.path.join(args.output_path, names[counters - 1])
        print("output shape", output.shape)
        write_nc.writenc(writepath, outnc)
        print("已输出第%d轮图片" % counters)

        output_images.extend(output.tolist())
    # 输出测试图片
    print("Total RMSE Loss is:", total_loss / counters)
    output_numpy_path = os.path.join(args.output_numpy_path, "prediction.npy")
    print("output_images shape", np.array(output_images).shape)
    np.save(output_numpy_path, np.array(output_images))
    print("testing end-----------------------------")


# 测试模型
def test_model(args):
    names = read_nc.get_name(args.test_path, args.area)
    all_dimen = []
    time_dimen = read_nc.get_time_dimen(args.test_path)
    lat, lon = read_nc.get_lat_lon_dimen(args.test_path)
    all_dimen.append(time_dimen)
    all_dimen.append(lat)
    all_dimen.append(lon)
    # 获取训练数据集
    test_data, test_label = tool.get_numpy_data(args.numpy_test_path, "test", "")
    print("test_data shape", test_data.shape)
    print("test_label shape", test_label.shape)
    test_data = torch.from_numpy(test_data).float()
    test_label = torch.from_numpy(test_label).float()
    torch_dataset = Data.TensorDataset(test_data, test_label)
    # 将训练集转化为模型可加载的格式, config.batch_size
    test_loader = Data.DataLoader(dataset=torch_dataset, batch_size=1)
    # 加载模型
    model = args.model
    try:
        checkpoint = torch.load(args.ckpt)
        print("加载的模型为：", args.ckpt)
        model.load_state_dict(checkpoint['net'])
        history_epoch = checkpoint['epoch']
        print("已加载第%d次Epoch的模型" % history_epoch)
    except:
        print("未加载模型")
    # 使用GPU训练
    loss_func = torch.nn.MSELoss()
    if torch.cuda.is_available():
        print("使用GPU训练模型")
        model = model.cuda()
        loss_func = loss_func.cuda()
    # 不启用 BatchNormalization 和 Dropout
    model.eval()
    counters = 0
    total_loss = 0.
    output_images = []
    print("testing start-----------------------------")
    for batch_x, batch_y in test_loader:
        if torch.cuda.is_available():
            batch_x = Variable(batch_x.cuda())
            batch_y = Variable(batch_y.cuda())
        counters += 1
        output = model(batch_x)
        rmse_loss = torch.sqrt(loss_func(output, batch_y))
        total_loss += float(rmse_loss)

        output = output.cpu().detach().numpy()
        outnc = [i for i in all_dimen]
        outnc.append(output)
        writepath = os.path.join(args.output_path, names[counters-1])
        print("output shape", output.shape)
        write_nc.writenc(writepath, outnc)
        print("已输出第%d轮图片" % counters)

        output_images.extend(output.tolist())
    # 输出测试图片
    print("Total RMSE Loss is:", total_loss/counters)
    output_numpy_path = os.path.join(args.output_numpy_path, "prediction.npy")
    print("output_images shape", np.array(output_images).shape)
    np.save(output_numpy_path, np.array(output_images))
    print("testing end-----------------------------")


# 保存pth模型
def save_model(args, model, epoch, min_loss):
    out_path = "%s_epoch_%d.pth" % (args.model_name, epoch)
    out_path = os.path.join(args.ckpt, out_path)
    # 建立模型参数字典
    state = {"net": model.state_dict(), "epoch": epoch, "min_loss":min_loss}
    torch.save(state, out_path)


# 加载模型中最优的参数，即最新的.pth文件
def load_model(args):
    list_paths = os.listdir(args.ckpt)
    # 按时间排序
    list_paths.sort(key=lambda fn: os.path.getmtime(args.ckpt + "/" + fn))
    file_new = os.path.join(args.ckpt, list_paths[-1])
    return file_new


# 计算多层模型损失值
def get_multi_loss(model, LR_x, HR_y):
    first_hr, second_hr = model(LR_x)
    mid_y = scipy.ndimage.zoom(HR_y, (1, 1, model.mid_size_w/model.output_size_w, model.mid_size_j/model.output_size_j))
    mid_y = torch.from_numpy(mid_y).float()
    mid_y = Variable(mid_y, requires_grad=True)
    first_loss = model.loss_func(first_hr, mid_y)
    second_loss = model.loss_func(second_hr, HR_y)
    return first_loss, second_loss