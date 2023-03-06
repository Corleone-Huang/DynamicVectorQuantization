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
from data.medical_datasets.DRIVE import DRIVEDataset

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--timestep_respacing", type=str, default="")
    parser.add_argument("--per_mask_sample_image", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--sample_num", type=int, default=100)
    # parser.add_argument("--save_path", type=str, default="./temp")

    return parser

if __name__ == "__main__":
    Shanghai = pytz.timezone("Asia/Shanghai")
    now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M")

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    save_path = opt.model_path.replace(".ckpt", "") + "_" + now + "_steps-{}".format(opt.timestep_respacing)
    save_mask_path = save_path + "_sample-mask_number-{}/".format(opt.sample_num)
    save_path += "_sample_number-{}/".format(opt.sample_num)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_mask_path, exist_ok=True)

    # init and save configs
    configs = OmegaConf.load(opt.yaml_path)
    if opt.timestep_respacing is not "":
        configs["model"]["params"]["diffusion_config"]["timestep_respacing"] = opt.timestep_respacing
    # model
    model = instantiate_from_config(configs.model)
    state_dict = torch.load(opt.model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval().cuda()

    count = 0

    dataset = DRIVEDataset(
        split="train", resolution=256, flip_p=0., is_eval=False, normalize=True, with_vessel=True, with_mask=True, expand_dataset=1000
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=1)

    for _, data in enumerate(dataloader):
        mask = data["image"][:, 3:, :, :].cuda()

        for i in range(opt.batch_size):
            sample_mask1 = torchvision.transforms.ToPILImage()(mask[i, :3, :, :])
            sample_mask2 = torchvision.transforms.ToPILImage()(mask[i, 3:, :, :])
            sample_mask1.save("{}/sample_mask1_{}_{}.png".format(save_mask_path, opt.sample_num, count))
            sample_mask2.save("{}/sample_mask2_{}_{}.png".format(save_mask_path, opt.sample_num, count))
            count += 1

        for j in range(opt.per_mask_sample_image):
            ddim_sample = model.diffusion.ddim_sample_loop(
                model=model.model,
                shape=(opt.batch_size, 3, 256, 256),
                noise=None,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs={"mask": mask},
                device=None,
                progress=True,
                eta=0.0,
                mask=mask,
            )

            for batch_i in range(opt.batch_size):
                sample = ddim_sample[batch_i]

                sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                sample = torchvision.transforms.ToPILImage()(sample)
                sample.save("{}/sample_{}_{}_{}-per-{}.png".format(save_path, opt.sample_num, count, batch_i, j))