import torch
import os
from dataset.create_data import MyData
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
from model.UNet import Unet



def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()


    epoch = 100
    batch_size = 16

    train_dir = "/share/home/dq070/Real-Time/cityscapes/leftImg8bit_sequence/train"
    val_dir = "/share/home/dq070/Real-Time/cityscapes/leftImg8bit_sequence/train"
    train_data = MyData(train_dir)
    val_data = MyData(val_dir)

    UNet = Unet().to(device)
    num_params = sum(p.numel() for p in UNet.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    if ngpu > 1:
        UNet = nn.DataParallel(UNet)

    # UNet.load_state_dict(torch.load('./checkpoint/mlp.params'))
    # UNet.eval()

    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(UNet.parameters(), lr = 6e-05, momentum = 0.1)
    train_losses = []  # 存储训练损失
    for epo in range(0, epoch):
        # train_data.make_train_list()
        # val_data.make_train_list()
        epoch_train_losses = 0
        train_batch = torch.zeros(batch_size, 3, 512, 512)
        val_batch = torch.zeros(batch_size, 3, 512, 512)
        for batch_num in range(0, int(train_data.__len__()/batch_size)):
            for i in range(0, batch_size):
                train_data_idx = batch_num*batch_size + i
                val_data_idx = (batch_num*batch_size + i+1) % val_data.__len__()
                train_data_item = train_data[train_data_idx]
                val_data_item = val_data[val_data_idx]
                train_batch[i, :, :, :] = train_data_item
                val_batch[i, :, :, :] = val_data_item
                train_batch = train_batch.to(device)
                val_batch = val_batch.to(device)

                optimizer.zero_grad()
            # train_batch = train_batch.view(-1, 3, 512, 512)
            # val_batch = val_batch.view(-1, 3, 512, 512)

                out = UNet(train_batch)
                loss = criterion(out, val_batch)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                epoch_train_losses += loss.item()  # 记录每个batch的总训练损失

            print('This is batch {i} in epoch {epo}, the loss is {loss}'.format(i=batch_num, epo=epo, loss=loss))
        torch.save(UNet.state_dict(), './checkpoint/mlp.params')
        epoch_train_losses /= train_data.__len__()
        train_losses.append(epoch_train_losses)

        # fig=plt.figure()
        plt.plot(range(0, epo+1), train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.savefig('/share/home/dq070/Real-Time/Zero_to_One/Unet_pretrain/AR-new/Loss_fig/loss_curve.png')

if __name__ == '__main__':
    train()