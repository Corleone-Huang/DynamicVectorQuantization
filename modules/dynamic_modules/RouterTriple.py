import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from modules.diffusionmodules.model import Normalize, nonlinearity

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(
            torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon
        )

class ResnetBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.norm1 = Normalize(channel)
        self.conv1 = torch.nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.norm2 = Normalize(channel)
        self.conv2 = torch.nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.conv2(h)
        return h + x

class TripleGrainEntropyRouter_v1(nn.Module):
    def __init__(self, channel=256, pixelnorm=False, scale_value=10.0):
        super().__init__()
        self.num_splits = 3
        self.pixelnorm = pixelnorm
        self.scale_value = scale_value
        if self.pixelnorm:
            self.norm = PixelNorm()
        else:
            self.norm = nn.Identity()
        self.conv_in = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0)
        self.resnet_block1 = ResnetBlock(channel)  # improve expressivity
        self.resnet_block2 = ResnetBlock(channel)  # improve expressivity
        self.gate = nn.Linear(channel, self.num_splits)
        self.scale = nn.Parameter(scale_value * torch.ones(1))  # increase selection difference, decrease the gap between training and sampling
        
    #     self.init_gate = 0.95
    #     self.init_parameters()
    
    # def init_parameters(self):
    #     num_splits = 3
    #     bias_value = math.log(math.sqrt(self.init_gate * (1 - num_splits) / (self.init_gate - 1)))
    #     trunc_normal_(self.gate.weight, std=.01)
    #     self.gate.bias.data[0:-1] = -bias_value / self.scale_value
    #     self.gate.bias.data[-1] = bias_value / self.scale_value

    
    def forward(self, h_fine=None, h_coarse=None, h_median=None, entropy=None):
        entropy = entropy.unsqueeze(1)
        entropy = self.norm(entropy)
        
        entropy = self.conv_in(entropy)
        entropy = self.resnet_block1(entropy)
        entropy = self.resnet_block2(entropy)
        entropy = entropy.permute(0,2,3,1)
        
        gate = self.gate(entropy) * self.scale

        return gate
    
class TripleGrainEntropyRouter_v2(nn.Module):
    def __init__(self, channel=256, pixelnorm=False, scale_value=10.0):
        super().__init__()
        self.num_splits = 3
        self.pixelnorm = pixelnorm
        self.scale_value = scale_value
        if self.pixelnorm:
            self.norm = PixelNorm()
        else:
            self.norm = nn.Identity()
        self.conv_in = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0)
        self.resnet_block1 = ResnetBlock(channel)  # improve expressivity
        self.resnet_block2 = ResnetBlock(channel)  # improve expressivity
        self.gate = nn.Linear(channel, self.num_splits)
        self.scale = nn.Parameter(scale_value * torch.ones(1))  # increase selection difference, decrease the gap between training and sampling

    def forward(self, h_fine=None, h_coarse=None, h_median=None, entropy=None):
        entropy = entropy.unsqueeze(1)
        entropy = self.norm(entropy)
        
        entropy = self.conv_in(entropy)
        entropy = self.resnet_block1(entropy)
        entropy = self.resnet_block2(entropy)
        entropy = entropy.permute(0,2,3,1)
        
        gate = self.gate(entropy) * self.scale

        return gate

class TripleGrainEntropyRouter(nn.Module):
    def __init__(self, channel=256, pixelnorm=False, scale_value=10.0):
        super().__init__()
        self.num_splits = 3
        self.pixelnorm = pixelnorm
        if self.pixelnorm:
            self.norm = PixelNorm()
        else:
            self.norm = nn.Identity()
        self.gate_pool = nn.AvgPool2d(2, 2)
        self.conv_in_0 = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0)
        self.resnet_block_0 = ResnetBlock(channel)  # improve expressivity
        self.gate_0 = nn.Linear(channel, 2)
        
        self.conv_in_1 = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0)
        self.resnet_block_1 = ResnetBlock(channel)  # improve expressivity
        self.gate_1 = nn.Linear(channel, 2)
        
        self.scale = nn.Parameter(scale_value * torch.ones(1))  # increase selection difference, decrease the gap between training and sampling

    def forward(self, h_fine=None, h_coarse=None, h_median=None, entropy=None):
        entropy_0, entropy_1 = entropy[0].unsqueeze(1), entropy[1].unsqueeze(1)
        entropy_0 = self.norm(entropy_0)  # 16x16
        entropy_1 = self.norm(entropy_1)  # 8x8
        
        entropy_0 = self.conv_in_0(entropy_0)
        entropy_0 = self.resnet_block_0(entropy).permute(0,2,3,1)
        gate_0 = self.gate_0(entropy) * self.scale
        
        entropy = torch.cat([entropy_0, self.gate_pool(entropy_1)], dim=1)
        
        entropy = self.conv_in(entropy)
        entropy = self.resnet_block1(entropy)
        entropy = self.resnet_block2(entropy)
        entropy = entropy.permute(0,2,3,1)
        
        gate = self.gate(entropy) * self.scale

        return gate