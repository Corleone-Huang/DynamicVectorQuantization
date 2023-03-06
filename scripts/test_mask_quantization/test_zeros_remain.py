from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import logging
import os
from omegaconf import OmegaConf

import torch
from tqdm import tqdm 
import sys
sys.path.append(os.getcwd())

from metrics.fid import get_inception_model
from metrics.fid import mean_covar_numpy, frechet_distance
from torch.utils.data import DataLoader
from utils.utils import instantiate_from_config

from data.ffhq import FFHQ
from data.imagenet_lmdb import Imagenet_LMDB
# from data.coco_t2i import CocoTrainValid
from data.mscoco import CocoPureImageTrainValid
from data.lsun import LSUNClass

import time

def setup_logger(result_path):
    log_fname = os.path.join(result_path, 'rfid.log')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s  -- special for random remain in MQVAE",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_fname), logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

if __name__ == '__main__':
    """
    Computes rFID, i.e., FID between val images and reconstructed images.
    Log is saved to `rfid.log` in the same directory as the given vqvae model. 
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size to use')
    parser.add_argument('--vqvae_config', type=str, default='', required=True,
                        help='vqvae_config path for recon FID')
    parser.add_argument('--vqvae_model', type=str, default='', required=True,
                        help='vqvae_model path for recon FID')
    parser.add_argument('--dataset_type', type=str, default='ffhq_val',
                        help='the corresponding dataset')
    parser.add_argument('--sleep', default=0, type=int)

    args = parser.parse_args()
    
    print("sleeping for {} seconds ...".format(args.sleep))
    time.sleep(args.sleep)
    print("end sleeping")
    
    result_path = os.path.dirname(args.vqvae_model)
    logger = setup_logger(result_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    config = OmegaConf.load(args.vqvae_config)
    model = instantiate_from_config(config.model)
    state_dict = torch.load(args.vqvae_model)['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    logger.info(f'vqvae config loaded from {args.vqvae_config}')
    logger.info(f'vqvae model loaded from {args.vqvae_model}')
    
    if args.dataset_type == "ffhq_train":
        dataset = FFHQ(split='train', resolution=256, is_eval=True)
    elif args.dataset_type == "ffhq_trainval":
        dataset = FFHQ(split='trainval', resolution=256, is_eval=True)
    elif args.dataset_type == "ffhq_val":
        dataset = FFHQ(split='val', resolution=256, is_eval=True)
    elif args.dataset_type == "imagenet_val":
        dataset = Imagenet_LMDB(split="val", resolution=256, is_eval=False)
    elif args.dataset_type == "coco_val":
        # dataset = CocoTrainValid(
        #     root="data/coco", split="valid", tok_name="bpe16k_huggingface", image_resolution=256, 
        #     transform_type="imagenet_val", context_length=32, is_eval=True, dropout=None
        # )
        dataset = CocoPureImageTrainValid(
            root="data/coco", split="valid", image_resolution=256, is_eval=True, transform_type="imagenet_val"
        )
    elif args.dataset_type == "lusn_church":
        dataset = LSUNClass(root="data/LSUN", category_name='church')
    elif args.dataset_type == "lusn_church_val":
        dataset = LSUNClass(root="data/LSUN", category_name='church_val')
    elif args.dataset_type == "lusn_bedroom":
        dataset = LSUNClass(root="data/LSUN", category_name='bedroom')
    elif args.dataset_type == "lusn_church_val":
        dataset = LSUNClass(root="data/LSUN", category_name='bedroom_val')
    else:
        raise NotImplementedError()
    logger.info('dataset: {}'.format(args.dataset_type))
    
    loader = DataLoader(
        dataset, shuffle=False, pin_memory=True, batch_size=args.batch_size, num_workers=16
    )
    
    # calculate rfid
    inception_model = get_inception_model().to(device)
    inception_model.eval()
    
    acts = []
    acts_recon = []

    sample_size_sum = 0.0
    sample_sum = torch.tensor(0.0, device=device)
    sample_sq_sum = torch.tensor(0.0, device=device)
    sample_max = torch.tensor(float('-inf'), device=device)
    sample_min = torch.tensor(float('inf'), device=device)
    
    # for i, batch in enumerate(loader):
    for batch in tqdm(loader, desc="compute acts"):
        # print(i)
        xs = model.get_input(batch, "image")
        xs = xs.to(device)
        
        # torchvision.utils.save_image(xs, "temp_original.png", normalize=True)
        
        # we are assuming that dataset returns value in -1 ~ 1 -> remap to 0 ~ 1
        xs = torch.clamp(xs*0.5 + 0.5, 0, 1)

        sample_sum += xs.sum()
        sample_sq_sum += xs.pow(2.0).sum()
        sample_size_sum += xs.numel()
        sample_max = max(xs.max(), sample_max)
        sample_min = min(xs.min(), sample_min)

        act = inception_model(xs).cpu().detach()
        acts.append(act)
        
        # here we assume that stage1 model input & output values are in -1 ~ 1 range
        # this may not cover DiscreteVAE
        imgs = 2. * xs - 1.

        # NOTE: modified here
        # xs_recon = torch.cat([
        #     model(imgs[i:i+1])[0] for i in range(imgs.shape[0])
        # ], dim=0)
        quant, diff, (_, _, quant_idx), masker_output = model.encode(imgs)
        sampled_length = masker_output["sampled_length"]
        sampled_quant = quant[:, :, :sampled_length]
        remain_quant = quant[:, :, sampled_length:]
        sample_index = masker_output["sample_index"]
        remain_index = masker_output["remain_index"]
        mask = masker_output["squeezed_mask"]

        sampled_quant_idx = quant_idx[:, :sampled_length]
        remain_quant_idx = quant_idx[:, sampled_length:]

        # remain_quant_idx_zeros = torch.zeros_like(remain_quant_idx).long().to(device)
        # remain_quant_zeros = model.quantize.get_codebook_entry(remain_quant_idx_zeros).permute(0, 2, 1)
        # xrec_zeros_remain = model.decode(sampled_quant, remain_quant_zeros, sample_index, remain_index, mask)

        remain_quant_idx_random = torch.randint(0, model.codebook_length, (remain_quant_idx.size())).long().to(device)
        remain_quant_random = model.quantize.get_codebook_entry(remain_quant_idx_random).permute(0, 2, 1)
        xrec_rand_remain = model.decode(sampled_quant, remain_quant_random, sample_index, remain_index, mask)

        xs_recon = xrec_rand_remain
        
        
        xs_recon = torch.clamp(xs_recon * 0.5 + 0.5, 0, 1)
        act_recon = inception_model(xs_recon).cpu().detach()
        acts_recon.append(act_recon)

    sample_mean = sample_sum.item() / sample_size_sum
    sample_std = ((sample_sq_sum.item() / sample_size_sum) - (sample_mean ** 2.0)) ** 0.5
    logging.info(f'val imgs. stats :: '
                f'max: {sample_max:.4f}, min: {sample_min:.4f}, mean: {sample_mean:.4f}, std: {sample_std:.4f}')


    acts = torch.cat(acts, dim=0)
    mu_acts, sigma_acts = mean_covar_numpy(acts)

    acts_recon = torch.cat(acts_recon, dim=0)
    mu_acts_recon, sigma_acts_recon = mean_covar_numpy(acts_recon.detach())
    
    rfid = frechet_distance(mu_acts, sigma_acts, mu_acts_recon, sigma_acts_recon)
    logger.info(f'rFID: {rfid:.4f}')
    logger.info(f'                            ')
    print("rfid: {}".format(rfid))