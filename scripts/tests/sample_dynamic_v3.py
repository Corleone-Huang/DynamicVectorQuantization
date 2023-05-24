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

def save_pickle(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--top_k_pos", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_p_pos", type=float, default=1.0)
    parser.add_argument("--sample_num", type=int, default=5000)

    return parser

if __name__ == "__main__":
    Shanghai = pytz.timezone("Asia/Shanghai")
    now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M-%S")

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    if opt.save_path is None:
        save_path = opt.model_path.replace(".ckpt", "") + "_" + now
        save_path_image = save_path + "_TopK-{}-{}_TopP-{}-{}_Temp-{}_Num-{}_image".format(opt.top_k, opt.top_k_pos, opt.top_p, opt.top_p_pos, opt.temperature, opt.sample_num)
        save_path_pickle = save_path + "_TopK-{}-{}_TopP-{}-{}_Temp-{}_Num-{}_pickle".format(opt.top_k, opt.top_k_pos, opt.top_p, opt.top_p_pos, opt.temperature, opt.sample_num)

        # save_path_image2 = save_path + "_TopK-{}-{}_TopP-{}-{}_Temp-{}_Num-{}_image".format(opt.top_k, opt.top_k_pos, opt.top_p, opt.top_p_pos, opt.temperature, opt.sample_num)
        # save_path_pickle2 = save_path + "_TopK-{}-{}_TopP-{}-{}_Temp-{}_Num-{}_pickle".format(opt.top_k, opt.top_k_pos, opt.top_p, opt.top_p_pos, opt.temperature, opt.sample_num)

        os.makedirs(save_path_image, exist_ok=True)
        os.makedirs(save_path_pickle, exist_ok=True)
        # os.makedirs(save_path_image2, exist_ok=True)
        # os.makedirs(save_path_pickle2, exist_ok=True)
    else:
        save_path = opt.save_path
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
    
    batch_size = opt.batch_size

    for i in trange(total_batch):
        if opt.sample_num % opt.batch_size != 0 and i == total_batch - 1:
            batch_size = opt.sample_num % opt.batch_size
            
        c_idx = 1026 * torch.ones(batch_size, 1).long().cuda()
        c_idx_pos = 258 * torch.ones(batch_size, 1).long().cuda()
        c_idx_relative = torch.zeros(batch_size, 1).long().cuda()
        c_idx_seg = torch.zeros(batch_size, 1).long().cuda()

        idx_sample, idx_pos_sample = model.sample_from_scratch(
            c_idx, c_idx_pos, c_idx_relative, c_idx_seg, 
            temperature = opt.temperature,
            sample = True,
            top_k = opt.top_k,
            top_p = opt.top_p,
            top_k_pos = opt.top_k_pos,
            top_p_pos = opt.top_p_pos,
            process = True,
        )

        sampled_images = model.decode_to_img(idx_sample, idx_pos_sample)
        sampled_images = sampled_images * 0.5 + 0.5
        sampled_images = torch.clamp(sampled_images, 0, 1)
        torchvision.utils.save_image(sampled_images, "{}/dynamic_v3_example_{}.png".format(save_path_image, i))

        save_pickle(
                os.path.join(save_path_pickle, 'samples_({}_{}).pkl'.format(i, total_batch)),
                sampled_images.cpu().numpy(),
            )