import torch
import os
from dataset.create_data import MyData
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
# import torch.utils.data
# import torch.utils.data.distributed
from model.UNet import Unet
import matplotlib.pyplot as plt
from model.ResNetAE_pytorch import ResNetAE
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training Img Restoration")
    parser.add_argument("--config", type=str, help="path to the yaml config file")
    args = parser.parse_args()
    return args

def load_config(filepath):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    return config


def evalutate(model, eval_data, device, batch_size):
    model.eval()   # evaluation
    criterion = torch.nn.MSELoss()
    eval_losses = 0

    with torch.no_grad():
        for batch_num in range(0, int(eval_data.__len__()/batch_size)):
            eval_input_batch = torch.zeros(batch_size, 3, 512, 512)
            eval_output_batch = torch.zeros(batch_size, 3, 512, 512)

            for i in range(0, batch_size):
                eval_input_data_idx = batch_num * batch_size + i
                eval_output_data_idx = (batch_num * batch_size + i + 1) % eval_data.__len__()
                eval_input_data_item = eval_data[eval_input_data_idx]
                eval_output_data_item = eval_data[eval_output_data_idx]
                eval_input_batch[i, :, :, :] = eval_input_data_item
                eval_output_batch[i, :, :, :] = eval_output_data_item

            eval_input_batch = eval_input_batch.to(device)
            eval_output_batch = eval_output_batch.to(device)
            out = model(eval_input_batch)
            loss = criterion(out, eval_output_batch)
            loss = loss.mean()
            eval_losses += loss.item()

        eval_losses /= eval_data.__len__() / batch_size
        print('Evaluation Loss: {:.4f}'.format(eval_losses))
        model.train()  # 切换模型回训练模式
        return eval_losses

def train(args):

    cfg = load_config(args.config)

    epoch = cfg["training"]["epoch"]
    batch_size = cfg["training"]["batch_size"]
    train_dir = cfg["data"]["train_dir"]
    val_dir = cfg["data"]["val_dir"]
    model_type = cfg["model"]["type"]
    weights_path = cfg["workdir"]["weights_path"]
    save_img_path = cfg["workdir"]["save_img_path"]
    learning_rate = cfg["training"]["learning_rate"]
    momentum = cfg["training"]["momentum"]
    cuda_visible_devices=cfg["cuda_visible_devices"]
    device = torch.device(cfg["training"]["device"])
    eval_interval = cfg["evaluation"]["interval"]
    eval_dir = cfg["data"]["eval_dir"]
    for path in [weights_path, save_img_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()

    #load imgs
    train_data = MyData(train_dir)
    val_data = MyData(val_dir)
    eval_data = MyData(eval_dir)

    # load model
    if model_type == 'resnet':
        net = ResNetAE().to(device)
    elif model_type == 'Unet':
        net = Unet().to(device)

    # whether parallel
    if ngpu > 1:
        net = nn.DataParallel(net)

    # Calculate Net Parameters
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # Define loss function
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=float(learning_rate), momentum=momentum)
    train_losses = []  #
    eval_epo = [] # store the evaluation epoch
    eval_losses_list = []  # initial eval loss to store different eval loss

    for epo in range(0, epoch):
        epoch_train_losses = 0 # initial epoch loss
        # initial train and val input
        train_batch = torch.zeros(batch_size, 3, 512, 512)
        val_batch = torch.zeros(batch_size, 3, 512, 512)
        for batch_num in range(0, int(train_data.__len__()/batch_size)):

            for i in range(0, batch_size):
                #define train and val index
                train_data_idx = batch_num*batch_size + i
                val_data_idx = (batch_num*batch_size + i+1) % val_data.__len__()
                # single train and val img
                train_data_item = train_data[train_data_idx]
                val_data_item = val_data[val_data_idx]
                # accumulate imgs
                train_batch[i, :, :, :] = train_data_item
                val_batch[i, :, :, :] = val_data_item
                # load imgs to GPU
                train_batch = train_batch.to(device)
                val_batch = val_batch.to(device)

            optimizer.zero_grad()
            out = net(train_batch)
            loss = criterion(out, val_batch)

            loss.backward()
            optimizer.step()
            epoch_train_losses += loss.item()  # per batch loss
            print('This is batch {i} in epoch {epo}, the loss is {loss}'.format(i=batch_num, epo=epo, loss=loss))

        # evaluation
        if (epo+1) % eval_interval == 0 or epo == epoch:
            eval_loss = evalutate(net, eval_data, device,batch_size)
            eval_losses_list.append(eval_loss)
            eval_epo.append(epo)


        # save weights
        torch.save(net.state_dict(), weights_path)
        epoch_train_losses /= train_data.__len__()/batch_size  # per epoch loss
        train_losses.append(epoch_train_losses)

        # loss fig
        epoches = range(0, epo+1)

        plt.figure()
        plt.plot(epoches, train_losses, label='Train loss')
        if (epo+1) % eval_interval == 0 or epo == epoch:
            plt.plot(eval_epo, eval_losses_list,'r-', label='Eval loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Eval Loss {model_type}')
        plt.legend()
        save_img = os.path.join(save_img_path, f"loss{model_type}.png")
        plt.savefig(save_img)

if __name__ == '__main__':
    args = parse_args()
    train(args)