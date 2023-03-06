import os
import sys
import numpy as np
from cv2 import normalize

sys.path.append(os.getcwd())
import argparse
import pickle

import torch
from omegaconf import OmegaConf
from tqdm import trange
from utils.utils import instantiate_from_config

import datetime
import pytz
from PIL import Image

import torchvision

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=100)
    # parser.add_argument("--save_path", type=str, default="./temp")

    return parser

if __name__ == "__main__":
    Shanghai = pytz.timezone("Asia/Shanghai")
    now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M")

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    save_path = opt.model_path.replace(".ckpt", "") + "_" + now
    save_path += "_sample_number-{}/".format(opt.sample_num)

    os.makedirs(save_path, exist_ok=True)

    # init and save configs
    configs = OmegaConf.load(opt.yaml_path)
    # model
    model = instantiate_from_config(configs.model)
    state_dict = torch.load(opt.model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval().cuda()

    model_kwargs = {} # uncondition

    # ddpm_sample_fn = (model.diffusion.p_sample_loop)
    # os.makedirs("temp/images_test", exist_ok=True)
    # torchvision.utils.save_image(ddpm_sample, "temp/images_test/ddpm_sample.png", normalize=False)
    # torchvision.utils.save_image(ddpm_sample, "temp/images_test/ddpm_sample_norm.png", normalize=True)

    count = 0

    while count < opt.sample_num:
        # ddpm_sample = ddpm_sample_fn(
        #     model.model,
        #     (opt.batch_size, 3, 256, 256),
        #     clip_denoised=False,
        #     model_kwargs=model_kwargs,
        #     progress=True
        # )
        ddpm_sample = model.diffusion.p_sample_loop(
            model=model.model,
            shape=(opt.batch_size, 3, 256, 256),
            noise=None,
            clip_denoised=False,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs={},
            device=None,
            progress=True,
            return_intermedia_results=True
        )
        # print(type(ddpm_sample))
        # print(len(ddpm_sample))
        # torchvision.utils.save_image(ddpm_sample[-1], "test.png", normalize=True)
        # exit()
        ddpm_sample = ddpm_sample[-1]

        for i in range(opt.batch_size):
            sample = ddpm_sample[i]

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = torchvision.transforms.ToPILImage()(sample)
            sample.save("{}/sample_{}_{}.png".format(save_path, opt.sample_num, count))
            # torchvision.utils.save_image(sample, "{}/sample_{}_{}.png".format(save_path, opt.sample_num, count), normalize=False)
            count += 1
        # print(len(ddpm_sample))
        # for i in [0, 500, 600, 700, 800, 900, 950, 999]:
        #     sample = ddpm_sample[i]
        #     print(sample.size())
        #     for j in range(sample.size(0)):
        #         sample_j = sample[j]
        #         sample_j = ((sample_j + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        #         sample_j = torchvision.transforms.ToPILImage()(sample_j)
        #         sample_j.save("{}/sample_{}_{}_{}.png".format(save_path, opt.sample_num, i, j))
        #         # torchvision.utils.save_image(sample, "{}/sample_{}_{}.png".format(save_path, opt.sample_num, count), normalize=False)
        
        # exit()

    # for i in range(opt.batch_size):
    #     sample = torchvision.transforms.ToPILImage()(ddpm_sample[i])
    #     print(sample)
    #     sample.save("temp/images_test/ddpm_sample_pil_{}.png".format(i))

    #     sample2 = ((ddpm_sample[i] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    #     sample2 = torchvision.transforms.ToPILImage()(sample2)
    #     print(sample2)
    #     sample2.save("temp/images_test/ddpm_sample_pil2_{}.png".format(i))