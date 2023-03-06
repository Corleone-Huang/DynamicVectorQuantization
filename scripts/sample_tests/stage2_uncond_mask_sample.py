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
import logging

def save_pickle(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, 
        default="")
    parser.add_argument("--model_path", type=str, 
        default="")
    parser.add_argument("--save_path", type=str,
        default=None)
    # # NOTE: 注意修改steps number
    # parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--hw", type=int, default=16)
    parser.add_argument("--mid_dim", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_k_pos", type=int, default=10)
    parser.add_argument("--sample_num", type=int, default=5000)
    
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
    now = datetime.datetime.now().astimezone(Shanghai).strftime("%m-%dT%H-%M-%S")

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    if opt.save_path is None:
        save_path = opt.model_path.replace(".ckpt", "") + "_" + now
        if not opt.debug:
            os.makedirs(save_path, exist_ok=True)
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
    sample_steps = model.first_stage_model.tokenizer.sample_num
    if not opt.debug:
        logging.info("sample_steps: {}".format(sample_steps))
    print("sample_steps: {}".format(sample_steps))

    for i in trange(total_batch):
        if opt.sample_num % opt.batch_size != 0 and i == total_batch - 1:
            batch_size = opt.sample_num % opt.batch_size
            
        z_start_indices = torch.randint(0, 1, (batch_size, 0)).cuda()
        c_indices = torch.ones(batch_size, 1) * model.sos_token
        c_indices = c_indices.long().cuda()
        
        z_pos_start_indices = torch.randint(0, 1, (batch_size, 0)).cuda()
        c_indices_pos = torch.ones(batch_size, 1) * model.sos_pos_token
        c_indices_pos = c_indices_pos.long().cuda()
        
        index_sample, pos_index_sample = model.sample(z_start_indices, c_indices, 
                                   z_pos_start_indices, c_indices_pos,
                                   steps=sample_steps,
                                   temperature=opt.temperature,
                                   sample=True,
                                   top_k=opt.top_k,
                                   top_k_pos=opt.top_k_pos,
                                   callback= lambda k: None)
        x_sample_nopix = model.decode_to_img(index_sample, pos_index_sample)
        
        # print(x_sample_nopix.size())
        # torchvision.utils.save_image(x_sample_nopix, "test.png", normalize=True)
        # torchvision.utils.save_image(x_sample_nopix, "test3.png", normalize=False)
        
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