import torch
import torch.nn as nn

import os, sys
sys.path.append(os.getcwd())

from utils.utils import instantiate_from_config
from torch.utils.data import DataLoader

from data.imagenet_lmdb import Imagenet_LMDB
from data.ffhq_lmdb import FFHQ_LMDB
from omegaconf import OmegaConf
import torchvision
from tqdm import tqdm
import argparse
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
# sns.set(style="darkgrid")

if __name__ == "__main__":
    hw = 16 # 32
    num = 256 // hw

    dataset = FFHQ_LMDB(split="train", resolution=256, is_eval=True)

    vqgan_config_path_f16 = "results/vqgan/vqgan_s1id00_ffhq_f16/configs/04-14T07-44-09-project.yaml"
    vqgan_model_path_f16 = "results/vqgan/vqgan_s1id00_ffhq_f16/checkpoints/last.ckpt"

    vqgan_config_path_f8 = "results/vqgan/vqgan_s1id01_ffhq_f8/configs/04-14T07-50-45-project.yaml"
    vqgan_model_path_f8 = "results/vqgan/vqgan_s1id01_ffhq_f8/checkpoints/last.ckpt"

    print("loading vqgan f16")
    vqgan_config_f16 = OmegaConf.load(vqgan_config_path_f16)
    vqgan_model_f16 = instantiate_from_config(vqgan_config_f16.model)
    vqgan_model_f16.load_state_dict(torch.load(vqgan_model_path_f16)["state_dict"])
    vqgan_model_f16 = vqgan_model_f16.cuda()

    print("loading vqgan f8")
    vqgan_config_f8 = OmegaConf.load(vqgan_config_path_f8)
    vqgan_model_f8 = instantiate_from_config(vqgan_config_f8.model)
    vqgan_model_f8.load_state_dict(torch.load(vqgan_model_path_f8)["state_dict"])
    vqgan_model_f8 = vqgan_model_f8.cuda()

    print("dataset length: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    with torch.no_grad():
        for data_i, data in tqdm(enumerate(dataloader)):
            images = data["image"].cuda()

            rec_f16, _ = vqgan_model_f16(images)
            rec_f8, _ = vqgan_model_f8(images)

            l1_loss_f16_list = []
            l1_loss_f8_list = []
            l1_loss_residual_list = []

            for i in range(num):
                for j in range(num):
                    local_image = images[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]

                    local_rec_f16 = rec_f16[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_f16 = nn.L1Loss()(local_rec_f16, local_image).item()

                    local_rec_f8 = rec_f8[:, :, hw*i:hw*(i+1), hw*j:hw*(j+1)]
                    l1_loss_f8 = nn.L1Loss()(local_rec_f8, local_image).item()

                    l1_residual_loss = abs(l1_loss_f16 - l1_loss_f8)

                    l1_loss_f16_list.append(l1_loss_f16)
                    l1_loss_f8_list.append(l1_loss_f8)
                    l1_loss_residual_list.append(l1_residual_loss)
            
            # if data_i == 200:
            #     break
    
    # print(l1_loss_residual_list)
    
    plt.hist(l1_loss_residual_list, bins=10, rwidth=0.9, density=True)

    # sns.kdeplot(l1_loss_residual_list)
    plt.savefig("test.png")