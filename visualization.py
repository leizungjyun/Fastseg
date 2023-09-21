import torch
from dataset.create_data import MyData
from model.UNet.UNet import Unet
from torchvision.utils import save_image
import os
import torchvision.transforms as transforms


def load_weights(UNet):
    pretrained_weights_path = '/share/home/dq070/Real-Time/Zero_to_One/Unet_pretrain/AR-new/checkpoint_HRDA/mlp.params'
    pretrained_dict = torch.load(pretrained_weights_path)
    UNet.load_state_dict(pretrained_dict, False)

def denorm(output_img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inverse_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    output_img = inverse_normalize(output_img)
    return output_img


def save_output_img(output_img, output_name):
    output_img = output_img.detach().clamp_(0, 1).cpu()
    output_img = output_img * 255
    # print(output_img.shape)
    # sys.exit()
    # print(len(output_img))
    # sys.exit()
    for i in range(output_img.shape[0]):
        output_img_new = output_img[i,...]
        # output_img_new = denorm(output_img_new)
        # output_img_new = np.transpose(output_img_new, (1,2,0))
        # output_img_new = np.array(output_img_new)
        # output_img_new= output_img_new.astype(np.uint8)
        path = '/share/home/dq070/Real-Time/Zero_to_One/Unet_pretrain/AR-new/visualize/RT01'
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
    UNet = Unet().to(device)
    UNet.eval()
    load_weights(UNet)
    output_batch = torch.zeros(batch_size, 3, 512, 512)
    for batch_num in range(0, int(input_img.__len__() / batch_size)):
        output_name = [['0']]*16
        # output_name = np.array(output_name)
        # print(output_name.shape)
        # sys.exit()
        for i in range(0, batch_size):
            output_batch[i, :, :, :] = input_img[batch_num*batch_size + i]
            output_batch = output_batch.to(device)
            output_img = UNet(output_batch)
            # print(output_img.type())
            # sys.exit()
            output_name[i][0]= input_img.get_file_name(batch_num*batch_size + i)
        save_output_img(output_img, output_name)
        # print(output_name)
        # sys.exit()

if __name__ == '__main__':
    visualize_output()