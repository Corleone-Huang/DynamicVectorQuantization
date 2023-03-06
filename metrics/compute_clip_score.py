# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os, sys
sys.path.append(os.getcwd())
import pickle
import argparse
import numpy as np
import torch
from torch.nn import functional as F

import clip
from PIL import Image
from tqdm import trange

# from metrics.fid import create_dataset_from_files

# from data_T2I.coco import CocoRawTextOnly

import glob

def get_clip():
    model_clip, preprocess_clip = clip.load("ViT-B/32", device='cpu')
    return model_clip, preprocess_clip


@torch.no_grad()
def clip_score(pixels, texts, model_clip, preprocess_clip, device=torch.device('cuda')):
    # pixels: 0~1 valued tensors
    pixels = pixels.cpu().numpy()
    pixels = np.transpose(pixels, (0, 2, 3, 1))

    images = [preprocess_clip(Image.fromarray((pixel*255).astype(np.uint8))) for pixel in pixels]
    images = torch.stack(images, dim=0).to(device=device)
    texts = clip.tokenize(texts).to(device=device)

    image_features = model_clip.encode_image(images)
    text_features = model_clip.encode_text(texts)

    scores = F.cosine_similarity(image_features, text_features).squeeze()

    return scores


def compute_clip_score(fake_path, device=torch.device('cuda'), ):

    model_clip, preprocess_clip = get_clip()
    model_clip.to(device=device)
    model_clip.eval()
    
    pkl_lists = glob.glob(os.path.join(fake_path, 'samples*.pkl'))
    
    # img_dataset = create_dataset_from_files(fake_path)
    # txt_dataset = create_dataset_from_files(text_path)

    # if dataset_name == 'coco':
    #     root = dataset_root if dataset_root else 'data/coco'
    #     txt_dataset = CocoRawTextOnly(
    #         root=root, split=split,  
    #         # tok_name=args.tok_name, image_resolution=args.image_resolution, 
    #         # transform_type=args.transform_type, context_length=args.context_length, 
    #         # is_eval=True
    #     )
    # else:
    #     raise ValueError(f'Unsupported dataset: {dataset_name}')

    # Here we assume that the order of imgs is same as the order of txts,
    # possibly has some duplicates at the end due to the distributed sampler.
    # assert len(img_dataset) >= len(txt_dataset)
    # img_dataset = torch.utils.data.Subset(img_dataset, np.arange(len(txt_dataset)))

    # img_loader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size)
    # txt_loader = torch.utils.data.DataLoader(txt_dataset, batch_size=batch_size)

    scores = []
    # for (imgs,), txts in zip(img_loader, txt_loader):
    for i in trange(len(pkl_lists)):
        with open(pkl_lists[i], 'rb') as f:
            # samples.append(pickle.load(f).cpu().numpy())
            imgs = pickle.load(f)
            if isinstance(imgs, np.ndarray):
                imgs = torch.from_numpy(imgs)
        
        text_pkl_name = pkl_lists[i].split("/")[-2] + "_Text"
        text_pkl_name2 = "text_" + pkl_lists[i].split("/")[-1]
        text_pkl_name_path = pkl_lists[i].replace(pkl_lists[i].split("/")[-2], text_pkl_name)
        text_pkl_name_path = text_pkl_name_path.replace(pkl_lists[i].split("/")[-1], text_pkl_name2)
        
        with open(text_pkl_name_path, 'rb') as f:
            # samples.append(pickle.load(f).cpu().numpy())
            txts = pickle.load(f)
                
        score = clip_score(imgs, txts, model_clip, preprocess_clip)
        scores.append(score.cpu().numpy())

    scores = np.concatenate(scores)
    scores_avg = scores.mean()

    return scores_avg


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--fake_path", type=str, default="")
    # parser.add_argument("--text_path", type=str, default="")
    # parser.add_argument("--dataset_name", type=str, default="coco")
    # parser.add_argument("--dataset_root", type=str, default="data/coco")
    # parser.add_argument("--split", type=str, default="valid")
    # parser.add_argument("--batch_size", type=int, default=10)
    
    # parser.add_argument("--tok_name", type=str, default="bpe16k_huggingface")
    # parser.add_argument("--image_resolution", type=int, default=256)
    # parser.add_argument("--transform_type", type=str, default="imagenet_val")
    # parser.add_argument("--context_length", type=str, default=32)
                       
    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    scores_avg = compute_clip_score(
        fake_path=opt.fake_path,
        device=torch.device('cuda'),
    )
    print("scores_avg: ", scores_avg)