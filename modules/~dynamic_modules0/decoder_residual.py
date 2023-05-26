import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.getcwd())

from modules.diffusionmodules.model import (AttnBlock, Normalize, ResnetBlock, Upsample, nonlinearity)

class DualGrainResidualDecoder(nn.Module):
    def __init__(
        self,
        *, 
        ch, 
        out_ch, 
        ch_mult=(1,2,4,8), 
        num_res_blocks,
        attn_resolutions, 
        resamp_with_conv=True, 
        in_channels,
        resolution, 
        z_channels, 
        dropout=0.0,
        give_pre_end=False, 
        only_add_fine_feature=True,
        **ignorekwargs
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.avg_pool_coarse = nn.AvgPool2d(2, 2)
        self.conv_in_coarse = torch.nn.Conv2d(z_channels, block_in, kernel_size=1, stride=1, padding=0)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

            if i_level == (self.num_resolutions - 1):
                self.conv_in_fine = torch.nn.Conv2d(z_channels, block_in, kernel_size=1, stride=1, padding=0)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

        self.only_add_fine_feature = only_add_fine_feature
    
    def forward(self, z, grain_indices=None):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        z_coarse = self.avg_pool_coarse(z)

        # z_coarse to block_in
        h = self.conv_in_coarse(z_coarse)
        z_fine = self.conv_in_fine(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
            
            if i_level == (self.num_resolutions - 1):
                if self.only_add_fine_feature:
                    assert grain_indices is not None
                    # 0 for coarse-grained and 1 for fine-grained
                    mask = grain_indices.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2).unsqueeze(1)
                    z_fine = mask * z_fine
                    h = h + z_fine
                else:
                    h = h + z_fine

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


if __name__ == "__main__":
    x = torch.randn(10, 256, 32, 32)
    grain_indices = torch.randint(0, 2, (10, 16, 16))

    model = DualGrainResidualDecoder(
        ch=128, 
        out_ch=3, 
        ch_mult=[1,1,2,2,4], 
        num_res_blocks=2,
        attn_resolutions=[8,16], 
        resamp_with_conv=True, 
        in_channels=256,
        resolution=256, 
        z_channels=256, 
        dropout=0.0,
        give_pre_end=False,
        only_add_fine_feature=False,
    )

    out = model(x, grain_indices)
    print(out.size())