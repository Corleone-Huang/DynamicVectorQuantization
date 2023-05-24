import os
import sys

sys.path.append(os.getcwd())
import argparse
import pickle

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import trange
from utils.utils import instantiate_from_config

import datetime
import pytz

import logging
import torchvision
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
    parser.add_argument("--hw", type=int, default=32)
    parser.add_argument("--mid_dim", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--sample_num", type=int, default=50000)
    
    parser.add_argument("--debug", action="store_true", help="debug or not")

    return parser

def setup_logger(result_path):
    log_fname = os.path.join(result_path, 'sample.log')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_fname), logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    Shanghai = pytz.timezone("Asia/Shanghai")
    now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M")

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    if opt.save_path is None:
        save_path = opt.model_path.replace(".ckpt", "") + "_" + now
        save_path += "_TopK-{}_TopP-{}_Temp-{}_Num-{}".format(opt.top_k, opt.top_p, opt.temperature, opt.sample_num)
        if not opt.debug:
            os.makedirs(save_path, exist_ok=True)
            print("create dirs!")
    else:
        save_path = opt.save_path
        if not opt.debug:
            os.makedirs(save_path, exist_ok=True)

    if not opt.debug:
        logger = setup_logger(save_path)
        logger.info(str(opt))

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
        if i == total_batch - 1:
            batch_size = opt.sample_num % opt.batch_size
            
        z_start_indices = torch.randint(0, 1, (batch_size, 0)).cuda()
        c_indices = torch.ones(batch_size, 1) * 0
        c_indices = c_indices.long().cuda()


        sample_code = model.transformer.sample(
            partial_sample=torch.randint(0, 1, (c_indices.size(0), model.permuter_config["params"]["rank1_hw"], model.permuter_config["params"]["rank1_hw"], int(model.permuter_config["params"]["rank2_hw"]**2))).to(model.device),
            model_aux=model.first_stage_model,
            cond=c_indices,
            start_loc=(0, 0),
            temperature=opt.temperature,
            top_k=opt.top_k,
            top_p=opt.top_p,
            is_tqdm=True,
            desc="Sampling",
            fast=True,
        )
        x_sample_nopix = model.decode_to_img(sample_code, (batch_size, opt.mid_dim, opt.hw, opt.hw))
        
        pixels = x_sample_nopix * 0.5 + 0.5
        pixels = torch.clamp(pixels, 0, 1)
        
        if not opt.debug:
            save_pickle(
                os.path.join(save_path, 'samples_({}_{}).pkl'.format(i, total_batch)),
                pixels.cpu().numpy(),
            )
        else:
            torchvision.utils.save_image(pixels, "temp/visual/{}.png".format(
                opt.yaml_path.split("/")[-3]
            ), normalize=False, nrow=4)
            exit()