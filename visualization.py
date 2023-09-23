import torch
from dataset.create_data import MyData
from model.UNet.UNet import Unet
from torchvision.utils import save_image
import cv2
import os
import torchvision.transforms as transforms
from model.transweather.transweather_model import Transweather_base
import numpy as np

def load_weights(model):
    pretrained_weights_path = '/share/home/dq070/Real-Time/Zero_to_One/Unet_pretrain/AR-new/checkpoint_transweather/transweather/epoch14_mlp.params'
    pretrained_dict = torch.load(pretrained_weights_path)
    model.load_state_dict(pretrained_dict, False)
    return model

def denorm(output_img):
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]
    inverse_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    output_img = inverse_normalize(output_img)
    return output_img


def save_output_img(output_img, output_name):
    output_img = output_img.detach().cpu()

    for i in range(output_img.shape[0]):
        output_img_new = output_img[i,...]
        output_img_new = denorm(output_img_new)
        # output_img = output_img * 255
        # output_img = np.clip(output_img, 0, 255)
        # output_img_new = np.transpose(output_img_new, (1,2,0))
        # output_img_new = np.array(output_img_new)
        # output_img_new= output_img_new.astype(np.uint8)
        path = '/share/home/dq070/Real-Time/Zero_to_One/Unet_pretrain/AR-new/checkpoint_vss/visual'
        if not os.path.exists(path):
            os.makedirs(path)
        output_path = os.path.join(path, output_name[i][0])
        save_image(output_img_new, output_path)
        # cv2.imwrite(output_path, output_img_new)
        print("Successively Save Img!")



def visualize_output():
    # 输入图像路径
    input_dir = '/share/home/dq070/hy-tmp/datasets/cityscapes/leftImg8bit/train-all-bright0.5'
    input_img = MyData(input_dir)
    batch_size = 16
    # load pretrained weights and instantiate model
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transweather_base().to(device)
    model.eval()
    model = load_weights(model)
    output_batch = torch.zeros(batch_size, 3, 224, 224)
    for batch_num in range(0, int(input_img.__len__() / batch_size)):
        output_name = [['0']]*16
        # output_name = np.array(output_name)
        # print(output_name.shape)
        # sys.exit()
        for i in range(0, batch_size):
            output_batch[i, :, :, :] = input_img[batch_num*batch_size + i]
            output_batch = output_batch.to(device)
            output_img = model(output_batch)
            # print(output_img.type())
            # sys.exit()
            output_name[i][0]= input_img.get_file_name(batch_num*batch_size + i)
        save_output_img(output_img, output_name)
        # print(output_name)
        # sys.exit()

if __name__ == '__main__':
    visualize_output()