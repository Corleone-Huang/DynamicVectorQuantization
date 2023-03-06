import warnings
warnings.filterwarnings("ignore")

import argparse
import glob
import os
import shutil
import sys

import lpips
import numpy as np
import torchvision
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.getcwd())

import logging

import torch
from data.ffhq import FFHQ
from data.imagenet_lmdb import Imagenet_LMDB
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import utils
from utils.utils import instantiate_from_config

from metrics.image_folder import make_dataset

parser = argparse.ArgumentParser(description='Image quality evaluations on the dataset')

parser.add_argument('--batch_size', type=int, default=4, help='Batch size to use')
parser.add_argument('--vqvae_config', type=str, default='', required=True, help='vqvae_config path for recon FID')
parser.add_argument('--vqvae_model', type=str, default='', required=True, help='vqvae_model path for recon FID')
parser.add_argument('--dataset_type', type=str, default='ffhq_val', help='the corresponding dataset')

parser.add_argument('--gt_path', type=str, default=None, help='path to original gt data')
parser.add_argument('--g_path', type=str, default=None, help='path to the generated data')
parser.add_argument('--save_path', type=str, default=None, help='path to save the best results')
parser.add_argument('--center', action='store_true', help='only calculate the center masked regions for the image quality')
parser.add_argument('--num_test', type=int, default=0, help='how many examples to load for testing')

args = parser.parse_args()
lpips_alex = lpips.LPIPS(net='alex')

def setup_logger(result_path):
    log_fname = os.path.join(result_path, 'stage-1-evaluation.log')
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


def calculate_score(img_gt, img_test):
    """
    function to calculate the image quality score
    :param img_gt: original image
    :param img_test: generated image
    :return: mae, ssim, psnr
    """

    l1loss = np.mean(np.abs(img_gt-img_test))

    psnr_score = psnr(img_gt, img_test, data_range=1)

    ssim_score = ssim(img_gt, img_test, multichannel=True, data_range=1, win_size=11)

    lpips_dis = lpips_alex(torch.from_numpy(img_gt).permute(2, 0, 1), torch.from_numpy(img_test).permute(2, 0, 1), normalize=True)

    return l1loss, ssim_score, psnr_score, lpips_dis.data.numpy().item()


if __name__ == '__main__':
    if args.dataset_type == "ffhq_val":
        dataset = FFHQ(split='val', resolution=256, is_eval=True)
    elif args.dataset_type == "imagenet_val":
        dataset = Imagenet_LMDB(split="val", resolution=256, is_eval=False)
    else:
        raise NotImplementedError()
    
    
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, pin_memory=True, batch_size=args.batch_size, num_workers=16
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    config = OmegaConf.load(args.vqvae_config)
    model = instantiate_from_config(config.model)
    state_dict = torch.load(args.vqvae_model)['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    if args.gt_path is None:
        gt_path = args.vqvae_model.replace(".ckpt", "_real")
        g_path = args.vqvae_model.replace(".ckpt", "_reconstruction")
    else:
        gt_path = args.gt_path
        g_path = args.g_path
        
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(g_path, exist_ok=True)
    
    # first sample original image and reconstrcution image
    count = 0
    for batch in tqdm(loader, desc="compute acts"):
        # print(i)
        xs = model.get_input(batch, "image")
        xs = xs.to(device)
        xs_recon = model(xs)[0]
        
        xs = torch.clamp(xs*0.5 + 0.5, 0, 1)
        xs_recon = torch.clamp(xs_recon * 0.5 + 0.5, 0, 1)
        for i in range(args.batch_size):
            torchvision.utils.save_image(
                xs[i], "{}/{}.png".format(gt_path, count), normalize=False
            )
            torchvision.utils.save_image(
                xs_recon[i], "{}/{}.png".format(g_path, count), normalize=False
            )
            count += 1
        
    gt_paths, gt_size = make_dataset(gt_path)
    g_paths, g_size = make_dataset(g_path)

    l1losses = []
    ssims = []
    psnrs = []
    lpipses = []

    size = args.num_test if args.num_test > 0 else gt_size
    
    if args.save_path == None:
        save_path = os.path.dirname(args.vqvae_model)
    else:
        save_path = args.save_path
        
    logger = setup_logger(save_path)
    
    logger.info(f'real loaded from {gt_path}')
    logger.info(f'reconstruction loaded from {g_path}')

    for i in range(size):
        gt_img = Image.open(gt_paths[i + 0*2000]).resize([256, 256]).convert('RGB')
        gt_numpy = np.array(gt_img).astype(np.float32) / 255.0
        if args.center:
            gt_numpy = gt_numpy[64:192, 64:192, :]

        l1loss_sample = 1000
        ssim_sample = 0
        psnr_sample = 0
        lpips_sample = 1000

        name = gt_paths[i + 0*2000].split('/')[-1].split(".")[0] + "*"
        g_paths = sorted(glob.glob(os.path.join(g_path, name)))
        num_files = len(g_paths)
        
        for j in range(num_files):
            index = j
            try:
                g_img = Image.open(g_paths[j]).resize([256, 256]).convert('RGB')
                g_numpy = np.array(g_img).astype(np.float32) / 255.0
                if args.center:
                    g_numpy = g_numpy[64:192, 64:192, :]
                l1loss, ssim_score, psnr_score, lpips_score = calculate_score(gt_numpy, g_numpy)
                if l1loss - ssim_score - psnr_score + lpips_score < l1loss_sample - ssim_sample - psnr_sample + lpips_sample:
                    l1loss_sample, ssim_sample, psnr_sample, lpips_sample = l1loss, ssim_score, psnr_score, lpips_score
                    best_index = index
            except:
                print(g_paths[index])

        if l1loss_sample != 1000 and ssim_sample !=0 and psnr_sample != 0:
            print(g_paths[best_index])
            print(l1loss_sample, ssim_sample, psnr_sample, lpips_sample)
            l1losses.append(l1loss_sample)
            ssims.append(ssim_sample)
            psnrs.append(psnr_sample)
            lpipses.append(lpips_sample)

            # if args.save_path is not None:
            #     utils.mkdir(args.save_path)
            #     shutil.copy(g_paths[best_index], args.save_path)

    logger.info('{:>10},{:>10},{:>10},{:>10}'.format('l1loss', 'SSIM', 'PSNR', 'LPIPS'))
    logger.info('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(np.mean(l1losses), np.mean(ssims), np.mean(psnrs), np.mean(lpipses)))
    logger.info('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(np.var(l1losses), np.var(ssims), np.var(psnrs), np.var(lpipses)))
    
    print('{:>10},{:>10},{:>10},{:>10}'.format('l1loss', 'SSIM', 'PSNR', 'LPIPS'))
    print('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(np.mean(l1losses), np.mean(ssims), np.mean(psnrs), np.mean(lpipses)))
    print('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(np.var(l1losses), np.var(ssims), np.var(psnrs), np.var(lpipses)))
