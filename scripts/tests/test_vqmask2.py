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
    model = model.eval().cuda()
    
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    val_dloader = data.val_dataloader()
    
    with torch.no_grad():
        for i, data in enumerate(val_dloader):
            images = data["image"].cuda()

            # dec, diff, masker_output = model(images)

            quant, diff, (_, _, quant_idx), masker_output = model.encode(images)

            sampled_length = masker_output["sampled_length"]
            sample_index = masker_output["sample_index"]
            remain_index = masker_output["remain_index"]
            squeezed_mask = masker_output["squeezed_mask"]

            sampled_quant_idx = quant_idx[:, :sampled_length]
            remain_quant_idx = quant_idx[:, sampled_length:]

            quant = model.post_quant_proj(quant.permute(0, 2, 1))

            sampled_quant = quant[:, :sampled_length, :]
            remain_quant = quant[:, sampled_length:, :]

            print(sampled_quant_idx)
            print(remain_quant_idx)
            exit()

            remain_quant_idx_zeros = torch.zeros_like(remain_quant_idx).long().cuda()
            remain_quant_zeros = model.quantize.get_codebook_entry(remain_quant_idx_zeros)
            remain_quant_zeros = model.post_quant_proj(remain_quant_zeros)

            sampled_quant_idx_zeros = torch.zeros_like(sampled_quant_idx).long().cuda()
            sampled_quant_zeros = model.quantize.get_codebook_entry(sampled_quant_idx_zeros)
            sampled_quant_zeros = model.post_quant_proj(sampled_quant_zeros)

            remain_quant_idx_random = torch.randint(0, model.codebook_length, (remain_quant_idx.size())).long().cuda()
            remain_quant_random = model.quantize.get_codebook_entry(remain_quant_idx_random)
            remain_quant_random = model.post_quant_proj(remain_quant_random)

            xrec = model.decoder(sampled_quant, remain_quant, sample_index, remain_index, squeezed_mask)
            xrec_zeros_remain = model.decoder(sampled_quant, remain_quant_zeros, sample_index, remain_index, squeezed_mask)
            xrec_zeros_sample = model.decoder(sampled_quant_zeros, remain_quant, sample_index, remain_index, squeezed_mask)
            xrec_rand_remain = model.decoder(sampled_quant, remain_quant_random, sample_index, remain_index, squeezed_mask)
            
            original_images = images * 0.5 + 0.5
            original_images = torch.clamp(original_images, 0, 1)
            xrec = xrec * 0.5 + 0.5
            xrec = torch.clamp(xrec, 0, 1)
            xrec_zeros_remain = xrec_zeros_remain * 0.5 + 0.5
            xrec_zeros_remain = torch.clamp(xrec_zeros_remain, 0, 1)
            xrec_zeros_sample = xrec_zeros_sample * 0.5 + 0.5
            xrec_zeros_sample = torch.clamp(xrec_zeros_sample, 0, 1)
            
            torchvision.utils.save_image(xrec, "temp/reconstruction.png", normalize=True, nrow=4)
            torchvision.utils.save_image(xrec_zeros_remain, "temp/reconstruction_zero_remain.png", normalize=True, nrow=4)
            torchvision.utils.save_image(xrec_zeros_sample, "temp/reconstruction_zero_sample.png", normalize=True, nrow=4)
            torchvision.utils.save_image(xrec_rand_remain, "temp/reconstruction_rand_remain.png", normalize=True, nrow=4)
            torchvision.utils.save_image(original_images, "temp/original.png", normalize=True, n_row=8)
            torchvision.utils.save_image(original_images * masker_output["binary_map"], "temp/binary_map.png", normalize=True, nrow=4)
            
            exit()