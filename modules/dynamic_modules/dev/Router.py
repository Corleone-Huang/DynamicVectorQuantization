import math
import torch
import torch.nn as nn
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

class MultipleGrainEntropyRouter(nn.Module):
    def __init__(self, num_splits, channel=256, pixelnorm=False, scale_value=10.0):
        super().__init__()
        self.num_splits = num_splits
        self.pixelnorm = pixelnorm
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

# class TripleGrainEntropyRouter(nn.Module):
#     def __init__(self, channel=256, pixelnorm=False, scale_value=10.0):
#         super().__init__()
#         self.num_splits = 3
#         self.pixelnorm = pixelnorm
#         if self.pixelnorm:
#             self.norm = PixelNorm()
#         else:
#             self.norm = nn.Identity()
#         self.gate_pool = nn.AvgPool2d(2, 2)
#         self.conv_in_0 = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0)
#         self.resnet_block_0 = ResnetBlock(channel)  # improve expressivity
#         self.gate_0 = nn.Linear(channel, 2)
        
#         self.conv_in_1 = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0)
#         self.resnet_block_1 = ResnetBlock(channel)  # improve expressivity
#         self.gate_1 = nn.Linear(channel, 2)
        
#         self.scale = nn.Parameter(scale_value * torch.ones(1))  # increase selection difference, decrease the gap between training and sampling

#     def forward(self, h_fine=None, h_coarse=None, h_median=None, entropy=None):
#         entropy_0, entropy_1 = entropy[0].unsqueeze(1), entropy[1].unsqueeze(1)
#         entropy_0 = self.norm(entropy_0)  # 16x16
#         entropy_1 = self.norm(entropy_1)  # 8x8
        
#         entropy_0 = self.conv_in_0(entropy_0)
#         entropy_0 = self.resnet_block_0(entropy).permute(0,2,3,1)
#         gate_0 = self.gate_0(entropy) * self.scale
        
        # entropy = torch.cat([entropy_0, self.gate_pool(entropy_1)], dim=1)
        
        # entropy = self.conv_in(entropy)
        # entropy = self.resnet_block1(entropy)
        # entropy = self.resnet_block2(entropy)
        # entropy = entropy.permute(0,2,3,1)
        
        # gate = self.gate(entropy) * self.scale

        # return gate

class DualGrainFeatureRouter(nn.Module):
    def __init__(self, num_channels, normalization_type="none", gate_type="1layer-fc"):
        super().__init__()
        self.gate_pool = nn.AvgPool2d(2, 2)
        self.gate_type = gate_type
        if gate_type == "1layer-fc":
            self.gate = nn.Linear(num_channels * 2, 2)
        elif gate_type == "2layer-fc-SiLu":
            self.gate = nn.Sequential(
                nn.Linear(num_channels * 2, num_channels * 2),
                nn.SiLU(inplace=True),
                nn.Linear(num_channels * 2, 2),
            )
        elif gate_type == "2layer-fc-ReLu":
            self.gate = nn.Sequential(
                nn.Linear(num_channels * 2, num_channels * 2),
                nn.ReLU(inplace=True),
                nn.Linear(num_channels * 2, 2),
            )
        else:
            raise NotImplementedError()

        self.num_splits = 2
        self.normalization_type = normalization_type
        if self.normalization_type == "none":
            self.feature_norm_fine = nn.Identity()
            self.feature_norm_coarse = nn.Identity()
        elif "group" in self.normalization_type:  # like "group-32"
            num_groups = int(self.normalization_type.split("-")[-1])
            self.feature_norm_fine = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
            self.feature_norm_coarse = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
        else:
            raise NotImplementedError()


    def forward(self, h_fine, h_coarse, entropy=None):
        h_fine = self.feature_norm_fine(h_fine)
        h_coarse = self.feature_norm_coarse(h_coarse)

        avg_h_fine = self.gate_pool(h_fine)

        # h_logistic = torch.cat([h_coarse, avg_h_fine], dim=1).permute(0,2,3,1)
        h_logistic = torch.cat([avg_h_fine, h_coarse], dim=1).permute(0,2,3,1)
        
        gate = self.gate(h_logistic)
        return gate
    

class DualGrainResidualFeatureEntropyRouter(nn.Module):
    def __init__(self, num_channels, normalization_type="none", gate_type="2layer-fc-SiLu"):
        super().__init__()
        self.gate_pool = nn.AvgPool2d(2, 2)
        self.gate_type = gate_type
        if gate_type == "2layer-fc-SiLu":
            self.gate = nn.Sequential(
                nn.Linear(num_channels * 2 + 1, num_channels * 2),
                nn.SiLU(inplace=True),
                nn.Linear(num_channels * 2, 2),
            )
        else:
            raise NotImplementedError()

        self.num_splits = 2
        self.normalization_type = normalization_type
        if self.normalization_type == "none":
            self.feature_norm_fine = nn.Identity()
            self.feature_norm_coarse = nn.Identity()
        elif "group" in self.normalization_type:  # like "group-32"
            num_groups = int(self.normalization_type.split("-")[-1])
            self.feature_norm_fine = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
            self.feature_norm_coarse = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
        else:
            raise NotImplementedError()


    def forward(self, h_fine, h_coarse, entropy=None):
        h_fine = self.feature_norm_fine(h_fine)
        h_coarse = self.feature_norm_coarse(h_coarse)

        avg_h_fine = self.gate_pool(h_fine)

        # h_logistic = torch.cat([h_coarse, avg_h_fine], dim=1).permute(0,2,3,1)
        # h_logistic = torch.cat([avg_h_fine, h_coarse], dim=1).permute(0,2,3,1)
        h_logistic = torch.cat([entropy.unsqueeze(1), avg_h_fine, h_coarse], dim=1).permute(0,2,3,1)
        
        gate = self.gate(h_logistic)
        return gate

# class MultipleGrainEntropyRouter(nn.Module):
#     def __init__(self, num_splits, init_gate=0.95):
#         super().__init__()
#         self.num_splits = num_splits
#         self.init_gate = init_gate
#         self.gate = nn.Linear(1, self.num_splits)
#         self.init_parameters()
        
#         self.mean = nn.Parameter(torch.zeros(1))
#         self.initted = False
        
#     def init_parameters(self):
#         trunc_normal_(self.gate.weight, std=.01)
#         if self.num_splits == 1:
#             nn.init.constant_(self.gate.bias.data, 0)
#         else:
#             bias_value = math.log(math.sqrt(self.init_gate * (1 - self.num_splits) / (self.init_gate - 1)))
#             self.gate.bias.data[-1] = bias_value  # initialize as finest granularity
#             self.gate.bias.data[:-1] = - bias_value
    
#     def init_value(self, entropy):
#         if self.initted:
#             return 
#         self.mean.data.copy_(entropy.mean())
#         self.initted = True

#     def forward(self, h_fine=None, h_coarse=None, h_median=None, entropy=None):
#         self.init_value(entropy)
#         entropy = entropy - self.mean
#         # entropy = entropy.unsqueeze(-1).repeat_interleave(self.num_splits, -1)
#         entropy = entropy.unsqueeze(-1)
#         gate = self.gate(entropy)

#         # for name, params in self.gate.named_parameters():
#         #     print(params.grad)
#         # print("---------------------------<end>--------------------------")
#         return gate

# class DualGrainHybridRouter(nn.Module):
#     def __init__(self, num_channels, normalization_type="none", gate_type="1layer-fc"):
#         super().__init__()
#         self.num_splits = 2
#         self.gate_pool = nn.AvgPool2d(2, 2)
#         self.gate_type = gate_type
#         if gate_type == "1layer-fc":
#             self.gate = nn.Linear(num_channels * 2, self.num_splits)
#         elif gate_type == "2layer-fc-SiLu":
#             self.gate = nn.Sequential(
#                 nn.Linear(num_channels * 2, num_channels * 2),
#                 nn.SiLU(inplace=True),
#                 nn.Linear(num_channels * 2, self.num_splits),
#             )
#         elif gate_type == "2layer-fc-ReLu":
#             self.gate = nn.Sequential(
#                 nn.Linear(num_channels * 2, num_channels * 2),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(num_channels * 2, self.num_splits),
#             )
#         else:
#             raise NotImplementedError()

#         self.num_splits = 2
#         self.normalization_type = normalization_type
#         if self.normalization_type == "none":
#             self.feature_norm_fine = nn.Identity()
#             self.feature_norm_coarse = nn.Identity()
#         elif "group" in self.normalization_type:  # like "group-32"
#             num_groups = int(self.normalization_type.split("-")[-1])
#             self.feature_norm_fine = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
#             self.feature_norm_coarse = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
#         else:
#             raise NotImplementedError()
        
#         self.entropy_gate = nn.Sequential(
#                 nn.Linear(1, 2 * self.num_splits),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(2 * self.num_splits, self.num_splits),
#             )

#         self.response_gate = nn.Parameter(torch.zeros(1)) 


#     def forward(self, h_fine, h_coarse, entropy):
#         h_fine = self.feature_norm_fine(h_fine)
#         h_coarse = self.feature_norm_coarse(h_coarse)

#         avg_h_fine = self.gate_pool(h_fine)

#         h_logistic = torch.cat([avg_h_fine, h_coarse], dim=1).permute(0,2,3,1)

#         gate = self.gate(h_logistic)

#         entropy_gate = self.entropy_gate(entropy.unsqueeze(-1))

#         return_gate = self.response_gate * gate + (1 - self.response_gate) * entropy_gate
        
#         return return_gate