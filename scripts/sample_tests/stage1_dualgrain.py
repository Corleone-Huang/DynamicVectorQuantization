import os
import sys

sys.path.append(os.getcwd())
import argparse

import torch
import torchvision
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config

from data.ffhq import FFHQ


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, 
        default="")
    parser.add_argument("--model_path", type=str, 
        default="")

    parser.add_argument("--batch_size", type=int, default=1)

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

    dset = FFHQ(split="val", resolution=256, is_eval=True)
    dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=False)

    for i,data in enumerate(dloader):
        image = data["image"].float().cuda()

        dec, diff, grain_indices = model(image)

        torchvision.utils.save_image(dec, "temp/test.png")
        print(grain_indices)
        exit()