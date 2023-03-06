# Sliced Wasserstein Distance (SWD)
import os, sys

import torch
sys.path.append(os.getcwd())

from scripts_diffusion.swd import swd
from data.medical_datasets.ChestXray import ChestXrayDataset_forMetrics
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from tqdm import trange

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
    dataset = ChestXrayDataset_forMetrics(
        split="test", resolution=256, is_eval=True, flip_p=0.5, normalize=True
    )

    for i in trange(len(dataset)):
        if i == 96:
            break
        if i == 0:
            x1 = dataset.__getitem__(0).unsqueeze(0)
        else:
            x1 = torch.cat([x1, dataset.__getitem__(0).unsqueeze(0)], dim=0)
    
    generated_dataset = EvalDataset(
        path="NUM_100"
    )
    for i in trange(len(generated_dataset)):
        if i == 0:
            x2 = generated_dataset.__getitem__(0).unsqueeze(0)
        else:
            x2 = torch.cat([x2, generated_dataset.__getitem__(0).unsqueeze(0)], dim=0)
    
    print(x1.size(), x2.size())
    out = swd(x1, x2, device="cuda") # Fast estimation if device="cuda"
    print(out) # tensor(53.6950)