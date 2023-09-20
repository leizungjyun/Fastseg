import torch
from model.UNet import Unet
from dataset.create_data import MyData
import torch.nn as nn
import torch.optim as optim
import os


def validate():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 读取GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()

    UNet = Unet().to(device)
    UNet.eval()  # 设置模型为评估模式

    val_dir =''
    val_data = MyData(val_dir)
    batch_size = 128
    val_losses = []  # 存储验证损失
    criterion = torch.nn.MSELoss()

    if ngpu > 1:
        UNet = nn.DataParallel(UNet)

    for batch_num in range(0, int(val_data.__len__() / batch_size)):
        val_batch = torch.zeros(batch_size, 3, 512, 512)
        for i in range(0, batch_size):
            val_data_idx = (batch_num * batch_size + i + 1) % val_data.__len__()
            val_data_item = val_data[val_data_idx]
            val_batch[i, :, :, :] = val_data_item
        val_batch = val_batch.to(device)

        with torch.no_grad():  # 在验证过程中不需要计算梯度
            out = UNet(val_batch)
            loss = criterion(out, val_batch)
            loss = loss.mean()

        val_losses.append(loss.item())  # 记录验证损失

        print('This is batch {i} in validation, the loss is {loss}'.format(i=batch_num, loss=loss))

    avg_val_loss = sum(val_losses) / len(val_losses)  # 计算平均验证损失

    # 可以根据需要记录或打印验证损失等指标
    print('Average validation loss:', avg_val_loss)

    # 可以返回验证损失或其他指标，或将其保存到文件中
    return avg_val_loss