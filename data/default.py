from easydict import EasyDict as edict

DefaultDataPath = edict()

DefaultDataPath.FFHQ = edict()
DefaultDataPath.FFHQ.train_lmdb = "/home/huangmq/AdaptiveVectorQuantization/data/FFHQ/FFHQ_lmdb/train"
DefaultDataPath.FFHQ.val_lmdb = "/home/huangmq/AdaptiveVectorQuantization/data/FFHQ/FFHQ_lmdb/val"

DefaultDataPath.FFHQ.root = "/home/huangmq/Datasets/FFHQ"

DefaultDataPath.CelebAHQ = edict()
DefaultDataPath.CelebAHQ.root = "/home/huangmq/Datasets/CelebA/CelebAHQ/CelebA-HQ"

DefaultDataPath.ImageNet = edict()

DefaultDataPath.ImageNet.root = "/home/huangmq/Datasets/ImageNet"
DefaultDataPath.ImageNet.train_write_root = "/home/huangmq/Datasets/ImageNet/train"
DefaultDataPath.ImageNet.val_write_root = "/home/huangmq/Datasets/ImageNet/val"