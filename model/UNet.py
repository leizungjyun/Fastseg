import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


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


class Unet(nn.Module):
    def __init__(self, useBN=False):
        super(Unet, self).__init__()

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

    def forward(self, x, save_path=None):
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
        # if save_path is not None:
        #     # 将 conv0_out 转换为图像并保存
        #     conv0_out_img = conv0_out.detach().clamp_(0, 1).cpu()  # 将张量移动到 CPU 上，并剪裁到 0 到 1 的范围内
        #     save_image(conv0_out_img, save_path)
        return conv0_out