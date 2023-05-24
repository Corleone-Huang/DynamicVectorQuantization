import torch
import torch.nn as nn
from einops import rearrange

class RankPermuter(nn.Module):
    def __init__(self, rank1_hw, rank2_hw):
        super().__init__()
        self.hw1 = rank1_hw
        self.hw2 = rank2_hw
    def forward(self, x, reverse=False):
        if not reverse:  # (B, HW) -> (B, H/n, W/n, n^2)
            return rearrange(x, "B (h1 h2 w1 w2) -> B h1 w1 (h2 w2)", h1=self.hw1, h2=self.hw2, w1=self.hw1, w2=self.hw2)
        else:
            return rearrange(x, "B h1 w1 (h2 w2) -> B (h1 h2 w1 w2)", h2=self.hw2, w2=self.hw2)



class RankDualGrainPermuter(nn.Module):
    def __init__(self, rank1_hw, rank2_hw, special_token_id=-1):
        super().__init__()
        self.hw1 = rank1_hw
        self.hw2 = rank2_hw
        self.special_token_id = special_token_id

    def forward(self, x, grain_indices=None, reverse=False, keep=True):
        """
        x: (B, HW)
        grain_indices: (B, H, W)
        """
        # replace the repeated coarse-grain ids into the special token id
        # 0 for coarse-grained and 1 for fine-grained
        if keep: 
            if not reverse:  # (B, HW) -> (B, H/n, W/n, n^2)
                return rearrange(x, "B (h1 h2 w1 w2) -> B h1 w1 (h2 w2)", h1=self.hw1, h2=self.hw2, w1=self.hw1, w2=self.hw2)
            else:
                return rearrange(x, "B h1 w1 (h2 w2) -> B (h1 h2 w1 w2)", h2=self.hw2, w2=self.hw2)
        else:
            if not reverse:  # (B, HW) -> (B, H/n, W/n, n^2)
                x = rearrange(x, "B (h1 h2 w1 w2) -> B h1 w1 (h2 w2)", h1=self.hw1, h2=self.hw2, w1=self.hw1, w2=self.hw2)
                special_matrix = self.special_token_id * torch.ones_like(x).to(x.device)
                x2 = torch.where(grain_indices.unsqueeze(-1)==0, special_matrix, x)
                x2 = torch.cat(
                    [x[:, :, :, :1], x2[:, :, :, 1:]], dim=-1
                )
                return x2
            else:
                repeated_first_matrix = x[:, :, :, :1].repeat_interleave(self.hw2 * self.hw2, dim=-1)
                grain_matrix = (x[:, :, :, 1:2] == self.special_token_id)
                x2 = torch.where(grain_matrix, repeated_first_matrix, x)

                return rearrange(x2, "B h1 w1 (h2 w2) -> B (h1 h2 w1 w2)", h2=self.hw2, w2=self.hw2)

if __name__ == "__main__":
    x1 = torch.randint(0, 1024, (1, 64))
    x2 = 4 * torch.ones_like(x1)

    grain_indices = torch.randint(0, 2, (1, 16)).view(1, 4, 4)
    grain_indices_repeat = grain_indices.view(1, 4, 4).repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2).view(1, 64)
    
    x = x1 * grain_indices_repeat + x2 * (1 - grain_indices_repeat)

    # print(x.view(1, 8, 8))
    print(x)

    permuter = RankDualGrainPermuter(
        rank1_hw=4, rank2_hw=2, special_token_id=-1, image_resolution=256
    )

    y = permuter(x, grain_indices)
    print(y)

    y2 = permuter(y, reverse=True)
    print(y2)