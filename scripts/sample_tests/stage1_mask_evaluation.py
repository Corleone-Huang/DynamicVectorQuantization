import os
import sys

from cv2 import normalize

sys.path.append(os.getcwd())
import argparse

import torch
import torchvision
from data.mscoco import CocoImagesAndCaptionsValidation
from data.imagenet_lmdb import Imagenet_LMDB
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config
from modules.tokenizers.tools import build_score_image


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, 
        default="logs/03-26T21-19-43_vqmask-50_coco_f16_1024/configs/03-26T21-19-43-project.yaml")
    parser.add_argument("--model_path", type=str, 
        default="logs/03-26T21-19-43_vqmask-50_coco_f16_1024/checkpoints/last.ckpt")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="/home/huangmq/git_repo/AdaptiveVectorQuantization/temp/0728")
    parser.add_argument("--dataset_type", type=str, default="imagenet_val")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    os.makedirs(opt.save_path, exist_ok=True)

    # init and save configs
    configs = OmegaConf.load(opt.yaml_path)
    # model
    model = instantiate_from_config(configs.model)
    state_dict = torch.load(opt.model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval().cuda()

    if opt.dataset_type == "coco":
        dset = CocoImagesAndCaptionsValidation(size=256)
        dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=True)
    elif opt.dataset_type == "imagenet_val":
        dset = Imagenet_LMDB(split="val", resolution=256, is_eval=True)
        dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, num_workers=0, shuffle=True)

    with torch.no_grad():
        for i,data in enumerate(dloader):
            image = data["image"].float().cuda()
            # image = image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)

            dec, diff, preforward_dict = model(image)

            real_image = image
            rec_image = dec

            real_image = torch.clamp(real_image * 0.5 + 0.5, 0, 1)
            rec_image = torch.clamp(rec_image * 0.5 + 0.5, 0, 1)

            binary_image = real_image*preforward_dict["binary_map"]
            score_map = preforward_dict["score_map"]
            score_map = build_score_image(real_image, score_map, low_color="blue", high_color="red", scaler=0.9)

            for j in range(opt.batch_size):
                torchvision.utils.save_image(real_image[j], "{}/{}_{}_real.png".format(opt.save_path, i, j), normalize=False)
                torchvision.utils.save_image(rec_image[j], "{}/{}_{}_rec.png".format(opt.save_path, i, j), normalize=False)
                torchvision.utils.save_image(binary_image[j], "{}/{}_{}_binary.png".format(opt.save_path, i, j), normalize=True)
                torchvision.utils.save_image(score_map[j], "{}/{}_{}_score_map.png".format(opt.save_path, i, j), normalize=True)