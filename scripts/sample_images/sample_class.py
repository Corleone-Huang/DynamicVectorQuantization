# class conditional sample

import os
import sys

sys.path.append(os.getcwd())
import argparse
import pickle

import torch
from omegaconf import OmegaConf
from tqdm import trange
from utils.utils import instantiate_from_config

import datetime
import pytz

import torchvision

from data.imagenet_lmdb import Imagenet_LMDB

def save_pickle(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    
    parser.add_argument("--save_path", type=str, default=None)
    
    # mannually setting
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--hw", type=int, default=16)
    parser.add_argument("--mid_dim", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--sample_num", type=int, default=10)
    
    parser.add_argument("--debug", action="store_true", help="debug or not")

    return parser

if __name__ == "__main__":
    sampled_class_list = [i for i in range(1000)]
    
    Shanghai = pytz.timezone("Asia/Shanghai")
    now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M")

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    if opt.save_path is None:
        save_path = opt.model_path.replace(".ckpt", "") + "_" + now
        save_path += "_TopK-{}_TopP-{}_Temp-{}_Num-{}_Image".format(opt.top_k, opt.top_p, opt.temperature, opt.sample_num)
        if not opt.debug:
            os.makedirs(save_path, exist_ok=True)
    else:
        save_path = opt.save_path
        if not opt.debug:
            os.makedirs(save_path, exist_ok=True)
    
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
    for class_id in sampled_class_list:
        for data in trange(total_batch):
            c_indices = (torch.ones(opt.batch_size, 1) * class_id).cuda().long()
            z_start_indices = torch.randint(0, 1, (opt.batch_size, 0)).cuda()
   
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

            for j in range(opt.batch_size):
                torchvision.utils.save_image(
                    pixels[j], "{}/class_{}_image_{}.png".format(
                    save_path, class_id, j
                ), normalize=False)