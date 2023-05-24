import random
import torch
from torchvision.datasets import CocoCaptions, VisionDataset

import os, sys
sys.path.append(os.getcwd())

from data.tokenizers_factory import create_tokenizer
from data.create_transforms import create_transforms
from utils.utils import instantiate_from_config
from tokenizers.processors import TemplateProcessing



class MSCOCO2014TrainValid(VisionDataset):
    splits = {'valid', 'train'}
    def __init__(self, root, split, image_resolution, transform_type=None, is_eval=False, tokenizer_config=None):
        assert split in self.splits, f'{split} is not in {self.splits}'
        
        assert transform_type in {"dalle", "dalle-vqvae", "clip", "clip-dvae", "none", "imagenet_train", "imagenet_val"}
        transform = create_transforms(transform_type, image_resolution, split, is_eval)
        super().__init__(root, transform=transform)

        self.split = split
        if tokenizer_config is not None:
            self.with_tokenizer = True
            self.tokenizer = instantiate_from_config(tokenizer_config)
        else:
            self.with_tokenizer = False
            self.tokenizer = None

        if split == "valid":
            self.dataset = CocoCaptions(root=f'{self.root}/images/val2014', annFile=f'{self.root}/annotations/captions_val2014.json')
        else:
            self.dataset = CocoCaptions(root=f'{self.root}/images/train2014', annFile=f'{self.root}/annotations/captions_train2014.json')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, text = self.dataset[item]

        if self.transform:
            img = self.transform(img)

        # text = ' '.join(text)  # text is a list of sentences. Concat them.
        if self.split == 'train':
            rnd_txt = random.randint(0, len(text)-1)
            text = text[rnd_txt]
        else:
            text = text[0]
        
        if self.with_tokenizer:
            output = self.tokenizer.get_tokens(text)
            ids, mask = output["token"], output["mask"]
            if not isinstance(ids, torch.LongTensor):
                ids = torch.LongTensor(ids)
            ids = ids.squeeze(0)

            return {
                "image": img,
                "caption": ids,
                "raw_text": text,
                "mask": mask,
            }
        else:
            return {
                "image": img,
                "raw_text": text,
            }

class MSCOCO2014TrainValidwithTokenizer(VisionDataset):
    splits = {'valid', 'train'}
    def __init__(self, root, split, tok_name, image_resolution, transform_type=None, context_length=77, is_eval=False, dropout=None):
        assert split in self.splits, f'{split} is not in {self.splits}'
        assert tok_name in {"bert_huggingface", "gpt2_huggingface", "bpe16k_huggingface", "bpe30k_huggingface"}
        assert transform_type in {"dalle", "dalle-vqvae", "clip", "clip-dvae", "none", "imagenet_train", "imagenet_val"}
        transform = create_transforms(transform_type, image_resolution, split, is_eval)
        super().__init__(root, transform=transform)

        self.split = split
        self.tokenizer = create_tokenizer(tok_name, lowercase=True, dropout=dropout)
        self.context_length = context_length - 1 # for adding [SEP] token

        if split == "valid":
            self.dataset = CocoCaptions(root=f'{self.root}/images/val2014',
                                        annFile=f'{self.root}/annotations/captions_val2014.json')
        else:
            self.dataset = CocoCaptions(root=f'{self.root}/images/train2014',
                                        annFile=f'{self.root}/annotations/captions_train2014.json')

        self.tokenizer.add_special_tokens(["[PAD]"])  # already exist
        self.tokenizer.add_special_tokens(["[SEP]"])
        self.tokenizer.add_special_tokens(["[CLS]"])

        self.tokenizer.enable_padding(length=self.context_length, pad_id=self.tokenizer.token_to_id("[PAD]"))
        self.tokenizer.enable_truncation(max_length=self.context_length)
        
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A",
            # pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                # ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            ],
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, text = self.dataset[item]

        if self.transform:
            img = self.transform(img)

        # text = ' '.join(text)  # text is a list of sentences. Concat them.
        if self.split == 'train':
            rnd_txt = random.randint(0, len(text)-1)
            text = text[rnd_txt]
        else:
            text = text[0]

        output = self.tokenizer.encode(text)
        ids = output.ids
        ids.append(self.tokenizer.token_to_id("[SEP]"))
        if not isinstance(ids, torch.LongTensor):
            ids = torch.LongTensor(ids)

        return {
            "image": img,
            "caption": ids,
            "raw_text": text,
        }

if __name__ == "__main__":
    # test code
    token_name = ["bert_huggingface", "gpt2_huggingface", "bpe16k_huggingface", "bpe30k_huggingface"]
    dataset = MSCOCO2014TrainValidwithTokenizer(
        root="data/coco", split="valid", tok_name=token_name[0], image_resolution=256, 
        transform_type="imagenet_val", context_length=32, is_eval=False, dropout=None
    )

    dataset = MSCOCO2014TrainValid(
        root="data/coco", split="valid", image_resolution=256, 
        transform_type="imagenet_val", is_eval=False,
        tokenizer_config=None
        # tokenizer_config={
        #     "target": "modules.clip_text_encoder.my_tokenizer.my_tokenize.Tokenize",
        #     "params": {
        #         "context_length": 77,
        #         "add_start_and_end": True,
        #         "with_mask": True,
        #         "pad_value": 0,
        #         "clip_embedding": False,
        #         "tokenizer_config": {
        #              'target': 'modules.clip_text_encoder.clip.simple_tokenizer.SimpleTokenizer',
        #              'params':{
        #                 'end_idx': 49152 # 16384 fo DALL-E
        #                 },
        #         }
        #     }
        # },
        )

    data = dataset.__getitem__(0)
    print(data)
    print(dataset.__len__())

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0, shuffle=True)

    for i, data in enumerate(dataloader):
        print(data["caption"])
        print(data["raw_text"])
        
        import torchvision
        torchvision.utils.save_image(data["image"], "image.png", normalize=True)
        exit()