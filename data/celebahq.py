import os
import numpy as np
import albumentations
import glob
import torchvision
from torch.utils.data import Dataset

import os, sys
sys.path.append(os.getcwd())
from data.data_utils import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from data.default import DefaultDataPath

class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex

class CelebAHQTrain(FacesBase):
    def __init__(self, size):
        super().__init__()
        glob_pattern = os.path.join(DefaultDataPath.CelebAHQ.root, 'train/images', '*.jpg')
        paths = sorted(glob.glob(glob_pattern))
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = None
        
        transforms_ = [torchvision.transforms.ToTensor(),]
        self.transforms = torchvision.transforms.Compose(transforms_)
    
    def __getitem__(self, i):
        example = self.data[i]
        example["image"] = self.transforms(example["image"])
        return example

class CelebAHQValidation(FacesBase):
    def __init__(self, size):
        super().__init__()
        glob_pattern = os.path.join(DefaultDataPath.CelebAHQ.root, 'test/images', '*.jpg')
        paths = sorted(glob.glob(glob_pattern))
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = None
        
        transforms_ = [torchvision.transforms.ToTensor(),]
        self.transforms = torchvision.transforms.Compose(transforms_)
    
    def __getitem__(self, i):
        example = self.data[i]
        example["image"] = self.transforms(example["image"])
        return example
    
if __name__ == "__main__":
    dset_train = CelebAHQTrain(size=1024)
    dset_val = CelebAHQValidation(size=1024)
    print(len(dset_train), len(dset_val))
    image = dset_val.__getitem__(0)['image']
    torchvision.utils.save_image(image, "image.png", normalize=True)