import torch
import os, sys
sys.path.append(os.getcwd())
from metrics.fid import compute_statistics_dataset, compute_statistics_from_files, frechet_distance
from data.medical_datasets.ChestXray import ChestXrayDataset_forMetrics
from data.medical_datasets.KaggleLung import KaggleLungDataset_forMetrics
from data.medical_datasets.Retinopathy import RetinopathyDataset_forMetrics

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="chestxray")
    return parser

class EvalDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.img_paths = sorted([os.path.join(self.path, name) for name in os.listdir(self.path) if name.endswith('.png')])

        transforms_ = [
            transforms.ToTensor(),
            # transforms.Resize(resolution),
            # transforms.CenterCrop(resolution),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
        self.transforms = transforms.Compose(transforms_)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index, with_transform=True):
        path = self.img_paths[index]

        sample = Image.open(path)
        if not sample.mode == "RGB":
            sample = sample.convert("RGB")
        sample = np.array(sample).astype(np.uint8)
        sample = (sample / 127.5 - 1.0).astype(float)

        if self.transforms is not None and with_transform:
            sample = self.transforms(sample)
        return sample.float()

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.dataset == "chestxray":
        dataset = ChestXrayDataset_forMetrics(
            split="test", resolution=256, is_eval=True, flip_p=0., normalize=True
        )
    elif opt.dataset == "kagglelung":
        dataset = KaggleLungDataset_forMetrics(
            resolution=256, is_eval=True, flip_p=0., normalize=False
        )
    elif opt.dataset == "retino":
        dataset = RetinopathyDataset_forMetrics(
            split="all", resolution=256, flip_p=0., normalize=True
        )

    mu_ref, sigma_ref, _, _ = compute_statistics_dataset(
        dataset=dataset, batch_size=50, inception_model=None, stage1_model=None, device=torch.device('cuda'), skip_original=False
    )

    # train_dataset = ChestXrayDataset_forMetrics(
    #     split="trainval", resolution=256, is_eval=True, flip_p=0.5, normalize=True
    # )
    # mu_fake, sigma_fake, _, _ = compute_statistics_dataset(
    #     dataset=train_dataset, batch_size=50, inception_model=None, stage1_model=None, device=torch.device('cuda'), skip_original=False
    # )
    # fid = frechet_distance(mu_ref, sigma_ref, mu_fake, sigma_fake)
    # print("Real:", fid)  # 19.771550865816238

    generated_dataset = EvalDataset(
        path=opt.path
    )  # fake100: 101.42816529550689, fake500: 73.37291931433757
    mu_fake, sigma_fake, _, _ = compute_statistics_dataset(
        dataset=generated_dataset, batch_size=50, inception_model=None, stage1_model=None, device=torch.device('cuda'), skip_original=False
    )
    fid = frechet_distance(mu_ref, sigma_ref, mu_fake, sigma_fake)
    print("fake:", fid)

    # batch_size = 50
    # device = torch.device('cuda')

    # target_test_path = "/home/huangmq/medical_datasets/chest_xray/test"
    # generated_path = "/home/huangmq/medical_datasets/chest_xray/test"

    # mu_ref, sigma_ref, acts_ref = compute_statistics_from_files(
    #     target_test_path,
    #     batch_size=batch_size,
    #     device=device,
    #     return_acts=True,
    # )

    # mu_fake, sigma_fake, acts_fake = compute_statistics_from_files(
    #     generated_path,
    #     batch_size=batch_size,
    #     device=device,
    #     return_acts=True,
    # )