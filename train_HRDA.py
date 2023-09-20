import torch
import os
from dataset.create_data import MyData
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.distributed as distributed
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import transforms


def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU(),

        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU()
    )

# Up sampling module
def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )


class Net(nn.Module):
    def __init__(self, useBN=False):
        super(Net, self).__init__()

        self.conv1 = add_conv_stage(3, 32)
        self.conv2 = add_conv_stage(32, 64)
        self.conv3 = add_conv_stage(64, 128)
        self.conv4 = add_conv_stage(128, 256)
        self.conv5 = add_conv_stage(256, 512)

        self.conv4m = add_conv_stage(512, 256)
        self.conv3m = add_conv_stage(256, 128)
        self.conv2m = add_conv_stage(128, 64)
        self.conv1m = add_conv_stage(64, 32)

        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

    def forward(self, x):
        conv1_out = self.conv1(x)
        F.dropout(conv1_out,p=0.5)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        F.dropout(conv2_out,p=0.5)
        conv3_out = self.conv3(self.max_pool(conv2_out))
        F.dropout(conv3_out,p=0.5)
        conv4_out = self.conv4(self.max_pool(conv3_out))
        F.dropout(conv4_out,p=0.5)
        conv5_out = self.conv5(self.max_pool(conv4_out))
        F.dropout(conv5_out,p=0.5)

        conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
        conv4m_out = self.conv4m(conv5m_out)
        conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
        conv3m_out = self.conv3m(conv4m_out_)
        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)
        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)
        conv0_out = self.conv0(conv1m_out)
        # conv0_out += 0.3 * mv
        return conv0_out

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()


    epoch = 300
    batch_size = 128

    train_dir = "/share/home/dq070/hy-tmp/datasets/cityscapes/leftImg8bit/train-bright"
    val_dir = "/share/home/dq070/hy-tmp/datasets/cityscapes/leftImg8bit/train-all"
    train_data = MyData(train_dir)
    val_data = MyData(val_dir)

    UNet = Net().to(device)
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
                val_data_idx = batch_num*batch_size + i
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
            # train_losses.append(loss.item())
            # plt.plot(range(0, batch_num+1), train_losses)
            # plt.xlabel('Batch')
            # plt.ylabel('Loss')
            # plt.title('Training Loss Curve')
            # plt.show()
            # plt.savefig('/share/home/dq070/Real-Time/Zero_to_One/Unet_pretrain/AR-new/Loss_fig/loss_curve.png')

            print('This is batch {i} in epoch {epo}, the loss is {loss}'.format(i=batch_num, epo=epo, loss=loss))
        torch.save(UNet.state_dict(), './checkpoint_HRDA/mlp.params')
        epoch_train_losses /= train_data.__len__()
        train_losses.append(epoch_train_losses)

        # fig=plt.figure()
        plt.plot(range(0, epo+1), train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.savefig('/share/home/dq070/Real-Time/Zero_to_One/Unet_pretrain/AR-new/Loss_fig_HRDA/loss_curve.png')

if __name__ == '__main__':
    train()