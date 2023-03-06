import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
from cv2 import normalize
from tqdm import tqdm 
sys.path.append(os.getcwd())
import argparse

import torch
import torchvision
from data.ffhq_lmdb import FFHQ_LMDB
from data.imagenet_lmdb import Imagenet_LMDB
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str)
    parser.add_argument("--model_path", type=str)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_type", type=str, default="ffhq_val")

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

    if opt.dataset_type == "ffhq_val":
        dset = FFHQ_LMDB(split="val", resolution=256, is_eval=True)
    elif opt.dataset_type == "ffhq_train":
        dset = FFHQ_LMDB(split="train", resolution=256, is_eval=True)
    elif opt.dataset_type == "imagenet_val":
        dset = Imagenet_LMDB(split="val", resolution=256, is_eval=True)
    elif opt.dataset_type == "imagenet_train":
        dset = Imagenet_LMDB(split="train", resolution=256, is_eval=True)
    else:
        raise NotImplementedError()
    dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=True)
    print("len(dloader): ", len(dloader))

    coding_ratio_list = []
    coding_length_list = []
    with torch.no_grad():
        for i,data in tqdm(enumerate(dloader)):
            image = data["image"].float().cuda()
            quant, emb_loss, info, masker_output = model.encode(image)
            
            coding_ratio = masker_output["coding_ratio"].cpu().numpy().tolist()
            coding_length = masker_output["coding_length"].cpu().numpy().tolist()

            coding_ratio_list += coding_ratio
            coding_length_list += coding_length
            
    coding_ratio_numpy = np.array(coding_ratio_list)
    coding_length_numpy = np.array(coding_length_list)
    print(len(coding_ratio_list), coding_ratio_numpy.min(), coding_ratio_numpy.max(), coding_ratio_numpy.mean(), coding_ratio_numpy.var())
    print(len(coding_length_list), coding_length_numpy.min(), coding_length_numpy.max(), coding_length_numpy.mean(), coding_length_numpy.var())