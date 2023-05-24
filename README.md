# DynamicVectorQuantization (CVPR 2023 highlight) (working in progress)

Offical PyTorch implementation of our CVPR 2023 highlight paper "[Towards Accurate Image Coding: Improved Autoregressive Image Generation with Dynamic Vector Quantization](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Towards_Accurate_Image_Coding_Improved_Autoregressive_Image_Generation_With_Dynamic_CVPR_2023_paper.pdf)".

**TL;DR** For vector-quantization (VQ) based autoregressive image generation, we propose a novel *variable-length* coding to replace existing *fixed-length* coding, which brings an accurate & compact code representation for images and a natural *coarse-to-fine* autoregressive generation order. 

Our framework includes: (1) DynamicQuantization VAE (DQ-VAE) which encodes image regions into variable-length codes based on their information densities. (2) DQ-Transformer which thereby generates images autoregressively from coarse-grained (smooth regions with fewer codes) to fine-grained (details regions with
more codes) by modeling the position and content of codes in each granularity alternately, through a novel stackedtransformer architecture and shared-content, non-shared position input layers designs.


![image](assets/dynamic_framework.png)

# Requirements and Installation
Please run the following command to install the necessary dependencies.

```
conda env create -f environment.yml
```

# Data Preparation
## ImageNet


# Training and Evaluation of DQVAE



# Training and Evaluation of DQ-Transformer