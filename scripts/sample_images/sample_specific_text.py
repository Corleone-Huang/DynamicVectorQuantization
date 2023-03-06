import os
import sys

sys.path.append(os.getcwd())
import argparse
import pickle

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.utils import instantiate_from_config

import datetime
import pytz

import logging
import torchvision

import numpy as np

from data.mscoco import MSCOCO2014TrainValid

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="coco_val")
    
    # mannually setting
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--hw", type=int, default=16)
    parser.add_argument("--mid_dim", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--sample_num", type=int, default=5000)
    
    parser.add_argument("--debug", action="store_true", help="debug or not")
    
    
    # for text dataset
    parser.add_argument("--tok_name", type=str, default="bpe16k_huggingface")
    parser.add_argument("--image_resolution", type=int, default=256)
    parser.add_argument("--transform_type", type=str, default="imagenet_val")
    parser.add_argument("--context_length", type=int, default=32)

    return parser



if __name__ == "__main__":
    given_text = "a large passenger place on a runway prepares to taxi."

    Shanghai = pytz.timezone("Asia/Shanghai")
    now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M")

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    # create dataset
    if opt.dataset == "coco_val":
        dataset = MSCOCO2014TrainValid(
            root='data/coco', split="valid", 
            tok_name=opt.tok_name, image_resolution=opt.image_resolution, 
            transform_type=opt.transform_type, context_length=opt.context_length, is_eval=True
        )
    elif opt.dataset == "coco_train":
        dataset = MSCOCO2014TrainValid(
            root='data/coco', split="train", 
            tok_name=opt.tok_name, image_resolution=opt.image_resolution, 
            transform_type=opt.transform_type, context_length=opt.context_length, is_eval=True
        )
    else:
        raise NotImplementedError()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=8, shuffle=False)   

    # init and save configs
    configs = OmegaConf.load(opt.yaml_path)
    # model
    model = instantiate_from_config(configs.model)
    state_dict = torch.load(opt.model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval().cuda()

    if opt.sample_num % opt.batch_size == 0:
        total_batch = opt.sample_num // opt.batch_size
    else:
        total_batch = opt.sample_num // opt.batch_size + 1

    batch_i = 0
    with torch.no_grad():
        
        output = dataset.tokenizer.encode(given_text)
        ids = output.ids
        ids.append(dataset.tokenizer.token_to_id("[SEP]"))

        z_start_indices = torch.randint(0, 1, (opt.batch_size, 0)).cuda()

        c_indices = torch.from_numpy(np.array(ids)).cuda().long().unsqueeze(0).repeat_interleave(opt.batch_size, dim=0)
        print(c_indices.size())

        index_sample = model.sample(z_start_indices, c_indices,
                                        steps=opt.steps,
                                        temperature=opt.temperature,
                                        sample=True,
                                        top_k=opt.top_k,
                                        top_p=opt.top_p,
                                        callback= lambda k: None)
        x_sample_nopix = model.decode_to_img(
            index_sample, (opt.batch_size, opt.mid_dim, opt.hw, opt.hw)
        )

        pixels = x_sample_nopix * 0.5 + 0.5
        pixels = torch.clamp(pixels, 0, 1)

        torchvision.utils.save_image(pixels, "test.png", normalize=False)