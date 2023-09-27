import torch
from dataset.create_data import MyData
from model.UNet.UNet import Unet
from torchvision.utils import save_image
import cv2
import os
import torchvision.transforms as transforms
from model.transweather.transweather_model import Transweather_base
from model.Resnet.ResNetAE_pytorch import ResNetAE as Resnet
import numpy as np
from PIL import Image
import sys


def load_weights(model):
    pretrained_weights_path = '/share/home/dq070/Real-Time/Zero_to_One/Unet_pretrain/AR-new/checkpoint_vss/resnet/epoch170_mlp.params'
    pretrained_dict = torch.load(pretrained_weights_path)
    model.load_state_dict(pretrained_dict, False)
    return model

def denorm(img):
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    return img.mul(std).add(mean)
    # output_img = inverse_normalize(output_img)
    # return output_img


def save_output_img(output_img, output_name):
    # output_img = output_img.detach().cpu()

    for i in range(0,16):
                # out = out.detach().cpu().numpy()
        img = output_img.detach().cpu().numpy()
        img = img[i, :, :, :]
        img = np.transpose(img, (1, 2, 0))
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        img = (img*255).astype('uint8')
        img = Image.fromarray(img)
        img.save(f'/share/home/dq070/Real-Time/Zero_to_One/Unet_pretrain/AR-new/checkpoint_vss/visual/{output_name[i]}.png')
        print(f"successively save img{i}!")




    # for i in range(0, 16):
    #     output_img_new = output_img[i,...]
    #     output_img_new = torch.clamp(denorm(output_img_new), 0, 1)
    #     # output_img = output_img.detach().cpu()

    #     path = '/share/home/dq070/Real-Time/Zero_to_One/Unet_pretrain/AR-new/checkpoint_vss/visual'
        
        
        
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     output_path = os.path.join(path, output_name[i][0])
    #     save_image(output_img_new, output_path)
    #     # cv2.imwrite(output_path, output_img_new)
    #     print("Successively Save Img!")



def visualize_output():
    # 输入图像路径
    input_dir = '/share/home/dq070/hy-tmp/datasets/cityscapes/leftImg8bit/train-all-bright0.5'
    input_img = MyData(input_dir)
    batch_size = 16
    # load pretrained weights and instantiate model
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transweather_base().to(device)
    model.eval()
    model = load_weights(model)
    output_batch = torch.zeros(batch_size, 3, 224, 224)
    for batch_num in range(0, int(input_img.__len__() / batch_size)):
        output_name = [['0']]*16
        for i in range(0, batch_size):
            output_batch[i, :, :, :] = input_img[batch_num*batch_size + i]
            output_batch = output_batch.to(device)
            output_img = model(output_batch)
            output_name[i][0]= input_img.get_file_name(batch_num*batch_size + i)
        save_output_img(output_img, output_name)
        # print(output_name)
        # sys.exit()

if __name__ == '__main__':
    visualize_output()