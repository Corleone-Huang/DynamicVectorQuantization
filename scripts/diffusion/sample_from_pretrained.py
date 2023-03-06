import torch
import torchvision
import os, sys
sys.path.append(os.getcwd())
import argparse

from diffusion.ADM.trainer_utils import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict, add_dict_to_argparser, 
)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=20,
        batch_size=4,
        use_ddim=False,
        model_path="diffusion/ADM_pretrained/256x256_diffusion_uncond.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    # path = "diffusion/ADM_pretrained/256x256_diffusion_uncond.pt"
    # state_dict = torch.load(path)
    # print(state_dict.keys())

    # MODEL_FLAGS=
    # --attention_resolutions 32,16,8 
    # --class_cond False 
    # --diffusion_steps 1000 
    # --image_size 256 
    # --learn_sigma True 
    # --noise_schedule linear 
    # --num_channels 256 
    # --num_head_channels 64 
    # --num_res_blocks 2 
    # --resblock_updown True 
    # --use_fp16 False
    # --use_scale_shift_norm True
    # python classifier_sample.py $MODEL_FLAGS --classifier_scale 10.0 
    # --classifier_path models/256x256_classifier.pt 
    # --model_path models/256x256_diffusion_uncond.pt 
    # $SAMPLE_FLAGS

    args = create_argparser().parse_args()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        torch.load(args.model_path)  # , map_location="cpu"
    )
    model.eval()
    model = model.cuda()

    model_kwargs = {}
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    sample = sample_fn(
        model,
        (args.batch_size, 3, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        progress=True,
        device="cuda",
    )
    # sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # sample = sample.permute(0, 2, 3, 1)
    # sample = sample.contiguous()

    print(sample.size())
    torchvision.utils.save_image(sample, "sample.png")
    torchvision.utils.save_image(sample, "sample_norm.png", normalize=True)