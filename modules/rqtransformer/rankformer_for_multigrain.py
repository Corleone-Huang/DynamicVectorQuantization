import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from collections import OrderedDict
from itertools import product
from tqdm import tqdm

from modules.autoregression.attentions import AttentionBlock
from modules.autoregression.utils import sample_from_logits

class DualRankformer(pl.LightningModule):
    def __init__(self, 
                 vocab_size_cond,
                 vocab_size,
                 block_size_cond,
                 block_size,  # (h,w,d)
                 body_config,
                 head_config,
                 shared_tok_emb,
                 stage1_embed_dim,
                 cumsum_depth_ctx,
                 apply_init=False,
                 use_differ_emb=False,  # use different token_emb for body and head
                 ):
        super().__init__()
        self.vocab_size_cond = vocab_size_cond
        self.vocab_size = vocab_size
        self.block_size_cond = block_size_cond
        self.block_size = block_size
        assert body_config["n_embd"] == head_config["n_embd"]
        assert body_config["embd_pdrop"] == head_config["embd_pdrop"]
        self.embed_dim = body_config["n_embd"]
        self.embd_pdrop = body_config["embd_pdrop"]
        self.shared_tok_emb = shared_tok_emb
        self.cumsum_depth_ctx = cumsum_depth_ctx
        self.use_differ_emb = use_differ_emb
        self.special_placeholder_id = vocab_size

        # ==== embedding layer definitions ====

        # vocab_size_cond == 1 => cond_emb works as a SOS token provider
        self.vocab_size_cond = max(self.vocab_size_cond, 1)
        self.block_size_cond = max(self.block_size_cond, 1)
        assert not (self.block_size_cond > 1 and self.vocab_size_cond == 1)
        self.cond_emb = nn.Embedding(self.vocab_size_cond, self.embed_dim)

        self.tok_emb, self.input_mlp, self.head_mlp = None, None, None
        if self.shared_tok_emb:
            self.body_input_mlp = nn.Linear(stage1_embed_dim, self.embed_dim)
            self.head_input_mlp = nn.Linear(stage1_embed_dim, self.embed_dim)
            self.special_placeholder_emb = nn.Parameter(torch.zeros(1, 1, 1, stage1_embed_dim))
        else:
            if self.use_differ_emb:
                self.body_tok_emb = nn.Embedding(self.vocab_size + 1, self.embed_dim, padding_idx=self.special_placeholder_id)
                self.head_tok_emb = nn.Embedding(self.vocab_size + 1, self.embed_dim, padding_idx=self.special_placeholder_id)
            else:
                self.tok_emb = nn.Embedding(self.vocab_size + 1, self.embed_dim, padding_idx=self.special_placeholder_id)

        self.pos_emb_cond = nn.Parameter(torch.zeros(1, self.block_size_cond, self.embed_dim))
        self.pos_emb_hw = nn.Parameter(torch.zeros(1, self.block_size[0] * self.block_size[1], self.embed_dim))
        self.pos_emb_d = nn.Parameter(torch.zeros(1, self.block_size[2], self.embed_dim))

        self.embed_drop = nn.Dropout(self.embd_pdrop, inplace=True)

        # ==== AR modeling layer definitions ====
        self.body_transformer = AttentionBlock(body_config)
        self.head_transformer = AttentionBlock(head_config)

        # ==== final fc layer definition ====
        self.classifier = nn.Sequential(OrderedDict([
            ('layer_norm', nn.LayerNorm(self.embed_dim)),
            (
                'linear',
                nn.Linear(self.embed_dim, self.vocab_size + 1)
            )
        ]))

        # init weights
        if apply_init:
            self.apply(self._init_weights)
        self.pos_emb_cond.data.normal_(mean=0.0, std=0.02)
        self.pos_emb_hw.data.normal_(mean=0.0, std=0.02)
        self.pos_emb_d.data.normal_(mean=0.0, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def embed_with_model_aux(self, xs, model_aux):
        xs_emb, _ = model_aux.get_code_emb_with_depth(xs)
        return xs_emb

    def forward(self, idx, idx_revised=None, c_idx=None, model_aux=None, return_loss=True):
        """
        idx_revised: replace the repeated coarse-grain indices in idx with the special placeholder idx
        """
        (B, H, W, D) = idx.shape

        idx = idx.reshape(B, H*W, D)
        idx_revised = idx_revised.reshape(B, H*W, D)
        if c_idx is None:
            c_idx = torch.zeros(B, self.block_size_cond, device=idx.device, dtype=torch.long)
        else:
            c_idx = c_idx.reshape(B, self.block_size_cond)

        seq_len = idx.shape[1]
        cond_len = c_idx.shape[1]

        # compute the embeddings for body
        if self.shared_tok_emb:
            xs_emb_stage1 = self.embed_with_model_aux(idx, model_aux)  # e.g., torch.Size([8, 256, 4, 256])
            xs_emb = torch.where(idx_revised.unsqueeze(-1)==self.special_placeholder_id, xs_emb_stage1, self.special_placeholder_emb)
            xs_emb = self.body_input_mlp(xs_emb)
        else:
            # if not share, we could directly input the idx_revised to get emb
            if self.use_differ_emb:
                xs_emb = self.body_tok_emb(idx_revised)
            else:
                xs_emb = self.tok_emb(idx_revised)
        
        conds_emb = self.cond_emb(c_idx) + self.pos_emb_cond[:, :cond_len, :]
        xs_emb = xs_emb.sum(dim=-2) + self.pos_emb_hw[:, :seq_len, :]
        latents = torch.cat(
            [
                conds_emb,
                xs_emb[:, :-1, :]
            ],
            dim=1,
        )
        # NOTE: dropout applied after everything is combined, not as before
        latents = self.embed_drop(latents)

        # body transformer
        latents = self.body_transformer(latents)
        spatial_ctx = latents[:, cond_len-1:]

        # compute the embeddings for head
        if self.shared_tok_emb:
            depth_ctx = self.embed_with_model_aux(idx, model_aux)

            if self.cumsum_depth_ctx:
                depth_ctx = torch.cumsum(depth_ctx, dim=-2)

            depth_ctx = self.head_input_mlp(depth_ctx)
        else:
            if self.use_differ_emb:
                depth_ctx = self.head_tok_emb(idx_revised)
            else:
                depth_ctx = self.tok_emb(idx_revised)

            if self.cumsum_depth_ctx:
                depth_ctx = torch.cumsum(depth_ctx, dim=-2)
        
        # NOTE: We are no longer applying spatial positional embedding to depth_ctx.
        # depth_ctx = depth_ctx + self.pos_emb_hw[:, :seq_len, :]

        depth_ctx_full = torch.cat(
            [
                spatial_ctx.view(B, seq_len, 1, -1),
                depth_ctx[:, :, :-1, :],
            ],
            dim=-2,
        )
        depth_ctx_full = depth_ctx_full.reshape(B * seq_len, D, -1)
        depth_ctx_full = depth_ctx_full + self.pos_emb_d[:, :D, :]

        # head transformer & final fc (classifier)
        head_outputs = self.head_transformer(depth_ctx_full)
        head_outputs = head_outputs.reshape(B, H, W, D, -1)

        seq_logits = self.classifier(head_outputs)

        if return_loss:
            # using idx_revised to get logistic
            loss = F.cross_entropy(seq_logits.view(-1, seq_logits.size(-1)), idx_revised.view(-1))
            return loss
        else:
            return seq_logits
    
    @torch.no_grad()
    def sample(self,
               partial_sample,
               model_aux=None,
               cond=None,
               start_loc=(0, 0),
               temperature=1.0,
               top_k=None,
               top_p=None,
               is_tqdm=True,
               desc="Sampling",
               fast=True,
               ):
        (H, W, D) = self.block_size

        if top_k is None:
            top_k_list = [self.vocab_size for i in range(D)]
        elif isinstance(top_k, int):
            top_k_list = [min(top_k, self.vocab_size) for i in range(D)]
        elif len(top_k) == 1:
            top_k_list = [min(top_k[0], self.vocab_size) for i in range(D)]
        else:
            top_k_list = [min(top_k[i], self.vocab_size) for i in range(D)]

        if top_p is None:
            top_p_list = [1.0 for _ in range(D)]
        elif isinstance(top_p, float):
            top_p_list = [min(top_p, 1.0) for _ in range(D)]
        elif len(top_p) == 1:
            top_p_list = [min(top_p[0], 1.0) for _ in range(D)]
        else:
            top_p_list = [min(top_p[i], 1.0) for i in range(D)]

        xs = partial_sample.clone()
        xs_revised = partial_sample.clone()
        assert xs.shape[1:] == torch.Size([H, W, D])

        sample_locs = list(product(range(H), range(W), range(D)))

        if is_tqdm:
            pbar = tqdm(sample_locs, total=len(sample_locs))
            pbar.set_description(desc)
        else:
            pbar = sample_locs
        
        for (h, w, d) in pbar:
            # the dual-grain-trigger should be activate when d==1 (if sample the special_placeholder token, then d=2/3 is no need to sample any more)
            if (h, w) < (start_loc[0], start_loc[1]):
                continue

            xs_partial = xs[:, :h + 1]
            xs_partial_revised = xs_revised[:, :h + 1]

            logits = self(idx=xs_partial, idx_revised=xs_partial_revised, c_idx=cond, model_aux=model_aux, return_loss=False)
            logits_hwd = logits[:, h, w, d]

            if d != 1:  # NOTE: turn off the prob of special_placeholder token when at the begining of sub-region
                logits_hwd = self.avoid_special_sampling(logits_hwd)
            
            _top_k = top_k_list[d]
            _top_p = top_p_list[d]

            samples_hwd = sample_from_logits(logits_hwd,
                                             temperature=temperature,
                                             top_k=_top_k,
                                             top_p=_top_p)

            sample_mask = (samples_hwd == self.special_placeholder_id)
            xs[:, h, w, d] = torch.where(sample_mask, xs[:, h, w, 0], samples_hwd)
            xs_revised[:, h, w, d] = samples_hwd

        return xs
    
    def avoid_special_sampling(self, logits):
        out = logits.clone()
        out[:, -1] = -float('Inf')  # avoid sample special_placeholder token
        return out
    
    def enforce_special_sampling(self, logits, mask):
        logits_clone = logits.clone()
        logits_clone[:, :-1] = -float('Inf')  # enforce sample special_placeholder token
        out = torch.where(mask.unsqueeze(-1), logits_clone, logits)
        return out