import io
import os
from pathlib import Path
import numpy as np
import lmdb
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import os, sys
sys.path.append(os.getcwd())
from data.default import DefaultDataPath

class Imagenet_LMDB(torchvision.datasets.VisionDataset):
    def __init__(self, split="train", resolution=256, is_eval=False, normalize=True, **kwargs):
        if split == "train":
            lmdb_path = DefaultDataPath.ImageNet.train_lmdb
        elif split == "val":
            lmdb_path = DefaultDataPath.ImageNet.val_lmdb
        else:
            raise ValueError()
        filelist_path = os.path.join(lmdb_path, "filelist.txt")

        root = str(Path(lmdb_path))
        super().__init__(root, **kwargs)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.stat()["entries"] // 2)
        
        with open(filelist_path, "r") as f:
            self.relpaths = f.read().splitlines()
        
        if split == "train" and not is_eval:
            if normalize:
                transforms_ = [
                    transforms.Resize(resolution),
                    transforms.RandomCrop(resolution),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            else:
                transforms_ = [
                    transforms.Resize(resolution),
                    transforms.RandomCrop(resolution),
                    transforms.ToTensor(),
                ]
        else:
            if normalize:
                transforms_ = [
                    transforms.Resize(resolution),
                    transforms.CenterCrop(resolution),
                    transforms.Resize((resolution, resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            else:
                transforms_ = [
                    transforms.Resize(resolution),
                    transforms.CenterCrop(resolution),
                    transforms.Resize((resolution, resolution)),
                    transforms.ToTensor(),
                ]
        self.transforms = transforms.Compose(transforms_)

    def __getitem__(self, index: int):
        image_key = self.relpaths[index].encode()
        class_key = (self.relpaths[index] + "class").encode()

        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(image_key)
            classbuf = txn.get(class_key)

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        # img.save("test.png")

        if self.transforms is not None:
            img = self.transforms(img)

        return {
            "image": img,
            "class_label": np.array(int(classbuf.decode()))
        }

    def __len__(self):
        return self.length




if __name__ == "__main__":
    from tqdm import trange
    imagenet_val = Imagenet_LMDB("val")
    print(len(imagenet_val))

    imagenet_train = Imagenet_LMDB("train")
    print(len(imagenet_train))