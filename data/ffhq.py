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

import os
from pathlib import Path

import torchvision
import torchvision.transforms as transforms


class ImageFolder(torchvision.datasets.VisionDataset):

    def __init__(self, root, train_list_file, val_list_file, 
                 split='train', resolution=256, is_eval=False, **kwargs):

        root = Path(root)
        super().__init__(root, **kwargs)

        self.train_list_file = train_list_file
        self.val_list_file = val_list_file

        self.split = self._verify_split(split)

        self.loader = torchvision.datasets.folder.default_loader
        self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS

        if self.split == 'trainval':
            fname_list = os.listdir(self.root)
            samples = [self.root.joinpath(fname) for fname in fname_list
                       if fname.lower().endswith(self.extensions)]
        else:
            listfile = self.train_list_file if self.split == 'train' else self.val_list_file
            with open(listfile, 'r') as f:
                samples = [self.root.joinpath(line.strip()) for line in f.readlines()]

        self.samples = samples
        
        if split == "train" and not is_eval:
            transforms_ = [
                transforms.RandomResizedCrop(resolution, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        else:
            transforms_ = [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        self.transforms = transforms.Compose(transforms_)

    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    @property
    def valid_splits(self):
        return 'train', 'val', 'trainval'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index, with_transform=True):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transforms is not None and with_transform:
            sample = self.transforms(sample)
        return {
            "image": sample
        }


class FFHQ(ImageFolder):
    train_list_file = Path(__file__).parent.joinpath('FFHQ/assets/ffhqtrain.txt')
    val_list_file = Path(__file__).parent.joinpath('FFHQ/assets/ffhqvalidation.txt')
    root = Path(__file__).parent.joinpath('FFHQ/FFHQ/')

    def __init__(self, split='train', resolution=256, is_eval=False, **kwargs):
        super().__init__(FFHQ.root, FFHQ.train_list_file, FFHQ.val_list_file, split, resolution, is_eval, **kwargs)
        
        
if __name__ == "__main__":
    dataset = FFHQ(split='train', resolution=256, is_eval=False)
    dataset_val = FFHQ(split='val', resolution=256, is_eval=False)
    
    print(len(dataset))
    print(len(dataset_val))
    
    # sample = dataset.__getitem__(0)
    
    # torchvision.utils.save_image(sample["image"], "sample_ffhq.png", normalize=True)