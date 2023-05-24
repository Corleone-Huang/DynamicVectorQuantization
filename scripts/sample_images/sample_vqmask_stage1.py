import os, sys

sys.path.append(os.getcwd())

from omegaconf import OmegaConf
import torch
import torchvision 
from utils.utils import instantiate_from_config
import argparse

from data.imagenet_lmdb import Imagenet_LMDB
from data.faceshq import FFHQ

from modules.tokenizers.tools import build_score_image

from tqdm import tqdm

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="imagenet")

    return parser

if __name__ == "__main__":
    vqgan_16x16_yaml_path = "results/vqgan/vqgan_s1id00_ffhq_f16/configs/04-14T07-44-09-project.yaml"
    vqgan_16x16_model_path = "results/vqgan/vqgan_s1id00_ffhq_f16/checkpoints/last.ckpt"

    pretrained_vqgan_16x16_yaml_path = "results/vqgan_pretrained/vqgan_imagenet_f16_1024/model.yaml"
    pretrained_vqgan_16x16_model_path = "results/vqgan_pretrained/vqgan_imagenet_f16_1024/last.ckpt"

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    yaml_path = opt.yaml_path
    model_path = opt.model_path

    os.makedirs("/home/huangmq/git_repo/AdaptiveVectorQuantization/temp/results/{}".format(opt.dataset), exist_ok=True)
    
    # init and save configs
    config = OmegaConf.load(yaml_path)
    
    # model
    model = instantiate_from_config(config.model)  # .cuda()
    model.load_state_dict(torch.load(model_path)["state_dict"], strict=False)
    model = model.cuda()

    config_vqgan = OmegaConf.load(vqgan_16x16_yaml_path)
    model_vqgan = instantiate_from_config(config_vqgan.model)
    model_vqgan.load_state_dict(torch.load(vqgan_16x16_model_path)["state_dict"], strict=False)
    model_vqgan = model_vqgan.cuda()

    config_vqgan_pretrained = OmegaConf.load(pretrained_vqgan_16x16_yaml_path)
    model_vqgan_pretrained = instantiate_from_config(config_vqgan_pretrained.model)
    model_vqgan_pretrained.load_state_dict(torch.load(pretrained_vqgan_16x16_model_path)["state_dict"], strict=False)
    model_vqgan_pretrained = model_vqgan_pretrained.cuda()
    
    if opt.dataset == "ffhq":
        dataset = FFHQ(split='train', resolution=256, is_eval=True)
    else:
        dataset = Imagenet_LMDB(split="train", resolution=256, is_eval=False)
    
    val_dloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size,
        num_workers=4, shuffle=True,
    )
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dloader)):
            images = data["image"].cuda()
            
            h = model.encoder(images)
            preforward_dict = model.tokenizer.preforward(h)
            h = preforward_dict["sample_h"]
            sample_index = preforward_dict["sample_index"]
            squeezed_mask = preforward_dict["squeezed_mask"]
            if "remain_index" in preforward_dict.keys():
                remain_index = preforward_dict["remain_index"]
            else:
                remain_index = None
            quant, emb_loss, info = model.quantize(h)
            postforward_dict = model.tokenizer.postforward(quant, sample_index, remain_index)
            quant = postforward_dict["decoder_embeeding"]
            xrec = model.decoder(x=quant, mask=squeezed_mask)
            
            
            original_images = images * 0.5 + 0.5
            original_images = torch.clamp(original_images, 0, 1)
            masked_images = original_images * preforward_dict["binary_map"]
            recon_images = xrec * 0.5 + 0.5
            recon_images = torch.clamp(recon_images, 0, 1)
            scored_images = build_score_image(images, preforward_dict["score_map"], scaler=0.8)

            recon_images_vqgan, _ = model_vqgan(images)
            recon_images_vqgan = torch.clamp(recon_images_vqgan * 0.5 + 0.5, 0, 1)

            recon_images_vqgan_pretrained, _ = model_vqgan_pretrained(images)
            recon_images_vqgan_pretrained = torch.clamp(recon_images_vqgan_pretrained * 0.5 + 0.5, 0, 1)
            
            for j in range(opt.batch_size):
                torchvision.utils.save_image(original_images[j], "temp/results/{}/{}_{}_original_images.png".format(opt.dataset,i,j))
                torchvision.utils.save_image(masked_images[j], "temp/results/{}/{}_{}_masked_images.png".format(opt.dataset,i,j))
                torchvision.utils.save_image(recon_images[j], "temp/results/{}/{}_{}_recon_images.png".format(opt.dataset,i,j))
                torchvision.utils.save_image(scored_images[j], "temp/results/{}/{}_{}_scored_images.png".format(opt.dataset,i,j))
                torchvision.utils.save_image(recon_images_vqgan[j], "temp/results/{}/{}_{}_vqgan.png".format(opt.dataset, i, j))
                torchvision.utils.save_image(recon_images_vqgan_pretrained[j], "temp/results/{}/{}_{}_vqgan_pretrained.png".format(opt.dataset, i, j))