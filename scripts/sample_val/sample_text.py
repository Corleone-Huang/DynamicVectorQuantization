# class conditional sample

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

from data_T2I.coco import CocoTrainValid

def save_pickle(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="coco_val")
    
    
    parser.add_argument("--save_path", type=str, default=None)
    
    # mannually setting
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--hw", type=int, default=16)
    parser.add_argument("--mid_dim", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=50)
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
        save_path_text = save_path + "_Text"
        if not opt.debug:
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(save_path_text, exist_ok=True)
    else:
        save_path = opt.save_path
        save_path_text = save_path + "_Text"
        if not opt.debug:
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(save_path_text, exist_ok=True)

    if not opt.debug:
        logger = setup_logger(save_path)
        logger.info(str(opt))
    
    # create dataset
    if opt.dataset == "coco_val":
        dataset = CocoTrainValid(
            root='data/coco', split="valid", 
            tok_name=opt.tok_name, image_resolution=opt.image_resolution, 
            transform_type=opt.transform_type, context_length=opt.context_length, is_eval=True
        )
    elif opt.dataset == "coco_train":
        dataset = CocoTrainValid(
            root='data/coco', split="train", 
            tok_name=opt.tok_name, image_resolution=opt.image_resolution, 
            transform_type=opt.transform_type, context_length=opt.context_length, is_eval=True
        )
    else:
        raise NotImplementedError()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True)    
        
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
    while True:
        for data in tqdm(dataloader):
            c_indices = data["caption"].cuda()
            z_start_indices = torch.randint(0, 1, (opt.batch_size, 0)).cuda()

            # print(c_indices.size(), z_start_indices.size())
            # exit()
   
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

            batch_i += 1
            if not opt.debug:
                save_pickle(
                    os.path.join(save_path, 'samples_({}_{}).pkl'.format(batch_i, total_batch)),
                    pixels.cpu().numpy(),
                )
                
                raw_texts = data["raw_text"]
                save_pickle(
                    os.path.join(save_path_text, 'text_samples_({}_{}).pkl'.format(batch_i, total_batch)),
                    raw_texts
                )
                
            else:
                torchvision.utils.save_image(pixels, "temp/visual/{}.png".format(
                    opt.yaml_path.split("/")[-3]
                ), normalize=False, nrow=4)
                exit()
            if batch_i >= total_batch:
                exit()