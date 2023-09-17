import os
import sys

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


class MyData(Dataset):
    def __init__(self, data_dir, mean=(128, 128, 128)):
        self.data_dir = data_dir
        # self.img_path = sorted(os.listdir(self.data_dir))
        self.img_paths = self.get_image_paths(data_dir)
        self.mean = mean

    def get_image_paths(self,data_dir):
        img_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    img_paths.append(os.path.join(root,file))
        return img_paths

    def __getitem__(self, idx,height=512, width=512):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        # print(img.size)
        # sys.exit()
        img = img.resize((width, height), Image.BICUBIC)
        transform = transforms.Compose([
            # 随机旋转图片
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # 正则化（降低模型复杂度）
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img =transform(img)


        img = np.array(img)
        # img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)
        return img

    def __len__(self):
        return len(self.img_paths)

    # def make_train_list(self, height=1024, width=512):
    #     self.data = torch.zeros(self.__len__(), 3, height, width)
    #     for i in range(0, self.__len__()):
    #         img = self.__getitem__(i)
    #         img = img.resize((width,height,3), Image.BICUBIC)
    #         img = np.array(img)
    #         img = np.transpose(img, (2, 0, 1))
    #         img = torch.tensor(img, dtype=torch.float32)
    #         # img -= torch.tensor(self.mean).view(3,1,1)
    #         self.data[i] = img
    #         # self.data[i, :, :, :] = torch.tensor(img[i])
    #     return data[i]
