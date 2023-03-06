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

from dataclasses import dataclass
import logging
from pathlib import Path

from omegaconf import OmegaConf

import os, sys
sys.path.append(os.getcwd())

from metrics.fid import compute_fid
from metrics.IS import compute_inception_score_from_files as compute_IS

DATASET_STATS_FOR_FID = {
    'imagenet': 'metrics/assets/fid_stats/imagenet_256_train.npz',
    'ffhq': 'metrics/assets/fid_stats/ffhq_256_train.npz',
    'ffhq_val': 'metrics/assets/fid_stats/ffhq_256_val.npz',
    'ffhq_trainval': 'metrics/assets/fid_stats/ffhq_256_trainval.npz',
    'lsun_bedroom': 'metrics/assets/fid_stats/lsun_256_bedroom.npz',
    'lsun_cat': 'metrics/assets/fid_stats/lsun_256_cat.npz',
    'lsun_church': 'metrics/assets/fid_stats/lsun_256_church.npz',
    'cc3m': 'metrics/assets/fid_stats/cc3m_256_val.npz',
    'coco_2014val': 'metrics/assets/fid_stats/coco_256_val.npz',
}

def compute_metrics(fake_path, ref_dataset, batch_size):
    results = {}

    ref_stat_path = DATASET_STATS_FOR_FID[ref_dataset]
    results['fid'] = compute_fid(fake_path, ref_stat_path, batch_size)

    if ref_dataset == 'imagenet':
        IS_mean, IS_std = compute_IS(fake_path)
        results['IS_mean'] = IS_mean
        results['IS_std'] = IS_std

    return results

if __name__ == '__main__':
    @dataclass
    class Arguments:
        fake_path: str
        ref_dataset: str
        batch_size: int

        @staticmethod
        def verify(args):
            datasets = set(DATASET_STATS_FOR_FID.keys())
            if args.ref_dataset not in datasets:
                raise ValueError(f"No dataset info found: {args.ref_dataset}")
    # print(Arguments)
    # exit()

    args = OmegaConf.structured(Arguments)
    args = OmegaConf.merge(args, OmegaConf.from_cli())  # type: Arguments

    Arguments.verify(args)

    log_path = Path(args.fake_path)
    log_path = os.path.join(log_path.parent, 'metrics.log')
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    
    logging.info('=' * 80)
    logging.info(f'{args}')
    
    results = compute_metrics(args.fake_path, args.ref_dataset, args.batch_size)
    print(results)

    logging.info('=' * 80)
    logging.info(results)