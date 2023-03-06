import os
import sys
import numpy as np

from cv2 import normalize

sys.path.append(os.getcwd())
import argparse

import torch
import torchvision
from data.mscoco import (CocoImagesAndCaptionsTrain,
                       CocoImagesAndCaptionsValidation)
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config

import torch.nn.functional as F


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, 
        default="logs/03-30T22-15-22_vqmask-50_coco_f16_1024-gpt/03-30T22-15-22-project.yaml")
    parser.add_argument("--model_path", type=str, 
        default="logs/03-30T22-15-22_vqmask-50_coco_f16_1024-gpt/last.ckpt")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--topk_pos", type=int, default=10)

    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # init and save configs
    configs = OmegaConf.load(opt.yaml_path)
    # model
    model = instantiate_from_config(configs.model)
    state_dict = torch.load(opt.model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval().cuda()

    dset = CocoImagesAndCaptionsValidation(size=256)
    dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=False)

    # total token numbers
    init_mask = torch.from_numpy(np.zeros(256)).float()

    for i, data in enumerate(dloader):

        x, c = model.get_xc(data, 4)
        x = x.to(device=model.device)
        c = c.to(device=model.device)

        quant_z, z_indices, z_indices_pos = model.encode_to_z(x)
        quant_c, c_indices, c_indices_pos = model.encode_to_c(c)

        # sample
        z_start_indices = z_indices[:, :0]
        z_pos_start_indices = z_indices_pos[:, :0]

        # sample
        index_sample, pos_index_sample = model.sample(
                                   z_start_indices, c_indices, z_pos_start_indices, c_indices_pos, 
                                   steps=model.first_stage_model.tokenizer.sample_num,
                                   temperature=opt.temperature,
                                   sample=True,
                                   top_k=opt.topk,
                                   top_k_pos=opt.topk_pos,
                                   callback= lambda k: None)
        x_sample_nopix = model.decode_to_img(index_sample, pos_index_sample)

        torchvision.utils.save_image(x_sample_nopix, "temp/sample_image.png", normalize=True)

        # height and weight
        init_mask = init_mask.cuda()
        for i in range(opt.batch_size):
            if i == 0:
                mask = init_mask.scatter(-1, pos_index_sample[i], 1.).view(16, 16).unsqueeze(0)
            else:
                mask_i = init_mask.scatter(-1, pos_index_sample[i], 1.).view(16, 16).unsqueeze(0)
                mask = torch.cat([mask, mask_i], dim=0)
        squeezed_mask = mask.view(opt.batch_size, -1)  # [batch_size, length]
        mask = F.interpolate(mask.float().unsqueeze(1), scale_factor=16, mode="nearest")

        torchvision.utils.save_image(mask, "temp/generate_mask.png", normalize=False)

        print(index_sample)
        print(pos_index_sample)
        exit()