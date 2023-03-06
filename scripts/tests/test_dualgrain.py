import sys, os
import torch
import torchvision
from torch.utils.data import DataLoader
sys.path.append(os.getcwd())
from PIL import Image
import numpy as np

from omegaconf import OmegaConf
import argparse
from utils.utils import instantiate_from_config

from data.ffhq import FFHQ
from data.imagenet_lmdb import Imagenet_LMDB

from modules.multigrained.utils import draw_dual_grain_256res, draw_dual_grain_256res_color
torch.set_printoptions(threshold=np.inf)

from modules.multigrained.permuter import DualGrainPermuter

def save_image(x, path):
    c,h,w = x.shape
    assert c==3
    x = ((x.detach().cpu().numpy().transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
    Image.fromarray(x).save(path)

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--yaml_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="ffhq_val")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    batch_size = opt.batch_size
    cfg = opt.yaml_path
    model_path = opt.model_path

    config = OmegaConf.load(cfg)

    # model
    model = instantiate_from_config(config.model).cuda()
    model.load_state_dict(torch.load(model_path)["state_dict"], strict=False)

    if opt.dataset == "ffhq_val":
        dataset = FFHQ(split='val', resolution=256, is_eval=True)
    elif opt.dataset == "imagenet_val":
        dataset = Imagenet_LMDB(split="train", resolution=256, is_eval=True)
    else:
        raise NotImplementedError()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    permuter = DualGrainPermuter(
        rank1_hw=16, 
        rank2_hw=2, 
        special_token_id=-1, 
        position_padding_value=256, 
        padding_value=1024
    )

    for i, data in enumerate(dataloader):
        # image = data["image"].cuda().permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        image = data["image"].cuda()

        quant, emb_loss, (perplexity, min_encodings, min_encoding_indices), grain_indices, gate = model.encode(image)

        rec = model.decode(quant)

        # print(min_encoding_indices)
        # print(grain_indices)

        # torchvision.utils.save_image(image, "temp/real/image_{}.png".format(i), normalize=True)
        # torchvision.utils.save_image(rec, "temp/fake/image_{}_rec.png".format(i), normalize=True)
        save_image(image[0], "temp/image_real_{}.png".format(i))
        save_image(rec[0], "temp/image_fake_{}.png".format(i))

        image_grain = draw_dual_grain_256res(images=image.clone(), indices=grain_indices)
        image_grain_color = draw_dual_grain_256res_color(images=image.clone(), indices=grain_indices, scaler=0.6)


        save_image(image_grain[0], "temp/image_grain_{}.png".format(i))
        save_image(image_grain_color[0], "temp/image_grain_color_{}.png".format(i))
        
        print(grain_indices.sum())

        out = permuter(
            indices=min_encoding_indices, grain_indices=grain_indices, position_indices=None, reverse=False
        )
        indices = out["indices"]
        position_indices = out["position_indices"]
        reproduced_indices = permuter(indices, grain_indices=None, position_indices=position_indices, reverse=True)
        reproduced_quant, _ = model.get_code_emb_with_depth(reproduced_indices)
        reproduced_rec = model.decode(reproduced_quant.permute(0, 3, 1, 2))
        save_image(reproduced_rec[0], "temp/image_fake_repoduce_{}.png".format(i))
        exit()