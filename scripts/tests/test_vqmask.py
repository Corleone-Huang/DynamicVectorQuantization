import os, sys

sys.path.append(os.getcwd())

from omegaconf import OmegaConf
import torch
import torchvision 
from utils.utils import instantiate_from_config
import argparse

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)

    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    yaml_path = opt.yaml_path
    model_path = opt.model_path
    
    # init and save configs
    config = OmegaConf.load(yaml_path)
    
    # model
    model = instantiate_from_config(config.model)  # .cuda()
    model.load_state_dict(torch.load(model_path)["state_dict"], strict=False)
    
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    val_dloader = data.val_dataloader()
    
    for i, data in enumerate(val_dloader):
        images = data["image"]  # .cuda()
        # images = data["image"].cuda().float().permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        dec, diff, preforward_dict = model(images)
        
        original_images = images * 0.5 + 0.5
        original_images = torch.clamp(original_images, 0, 1)
        masked_images = preforward_dict["binary_map"] * 0.5 + 0.5
        masked_images = torch.clamp(masked_images, 0, 1)
        
        
        torchvision.utils.save_image(dec, "temp/1.png", normalize=True)
        torchvision.utils.save_image(images, "temp/0.png", normalize=True)
        torchvision.utils.save_image(preforward_dict["binary_map"], "temp/2.png", normalize=True)
        
        exit()