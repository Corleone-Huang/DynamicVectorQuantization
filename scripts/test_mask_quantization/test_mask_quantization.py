import os
import sys

sys.path.append(os.getcwd())
import argparse

import torch
import torchvision
# from data.coco import CocoImagesAndCaptionsValidation
from data.mscoco import CocoPureImageTrainValid
from data.imagenet_lmdb import Imagenet_LMDB
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config
from modules.tokenizers.tools import build_score_image


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="/home/huangmq/git_repo/AdaptiveVectorQuantization/temp/1226")
    parser.add_argument("--dataset_type", type=str, default="coco_val")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    os.makedirs(opt.save_path, exist_ok=True)

    # init and save configs
    configs = OmegaConf.load(opt.yaml_path)
    # model
    model = instantiate_from_config(configs.model)
    state_dict = torch.load(opt.model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval().cuda()

    if opt.dataset_type == "coco_val":
        # dset = CocoImagesAndCaptionsValidation(size=256)
        dset = CocoPureImageTrainValid(
            root="data/coco", split="valid", image_resolution=256, is_eval=True, transform_type="imagenet_val"
        )
        dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=True)
    elif opt.dataset_type == "imagenet_val":
        dset = Imagenet_LMDB(split="val", resolution=256, is_eval=True)
        dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=True)

    with torch.no_grad():
        for i,data in enumerate(dloader):
            image = data["image"].float().cuda()
            
            rec, diff = model(image)

            torchvision.utils.save_image(torch.cat([image, rec], dim=0), "temp.png", normalize=True)
            exit()