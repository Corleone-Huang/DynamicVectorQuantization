import torch
import torch.nn as nn


class MultipleGrainEntropyRouter(nn.Module):
    def __init__(self, num_splits):
        super().__init__()
        self.num_splits = num_splits
        self.gate = nn.Sequential(
                nn.Linear(1, 2 * self.num_splits),
                nn.ReLU(inplace=True),
                nn.Linear(2 * self.num_splits, self.num_splits),
            )

    def forward(self, h_fine=None, h_coarse=None, h_median=None, entropy=None):
        gate = self.gate(entropy.unsqueeze(-1))
        return gate

class DualGrainHybridRouter(nn.Module):
    def __init__(self, num_channels, normalization_type="none", gate_type="1layer-fc"):
        super().__init__()
        self.num_splits = 2
        self.gate_pool = nn.AvgPool2d(2, 2)
        self.gate_type = gate_type
        if gate_type == "1layer-fc":
            self.gate = nn.Linear(num_channels * 2, self.num_splits)
        elif gate_type == "2layer-fc-SiLu":
            self.gate = nn.Sequential(
                nn.Linear(num_channels * 2, num_channels * 2),
                nn.SiLU(inplace=True),
                nn.Linear(num_channels * 2, self.num_splits),
            )
        elif gate_type == "2layer-fc-ReLu":
            self.gate = nn.Sequential(
                nn.Linear(num_channels * 2, num_channels * 2),
                nn.ReLU(inplace=True),
                nn.Linear(num_channels * 2, self.num_splits),
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
        
        self.entropy_gate = nn.Sequential(
                nn.Linear(1, 2 * self.num_splits),
                nn.ReLU(inplace=True),
                nn.Linear(2 * self.num_splits, self.num_splits),
            )

        self.response_gate = nn.Parameter(torch.zeros(1)) 


    def forward(self, h_fine, h_coarse, entropy):
        h_fine = self.feature_norm_fine(h_fine)
        h_coarse = self.feature_norm_coarse(h_coarse)

        avg_h_fine = self.gate_pool(h_fine)

        h_logistic = torch.cat([avg_h_fine, h_coarse], dim=1).permute(0,2,3,1)

        gate = self.gate(h_logistic)

        entropy_gate = self.entropy_gate(entropy.unsqueeze(-1))

        return_gate = self.response_gate * gate + (1 - self.response_gate) * entropy_gate
        
        return return_gate