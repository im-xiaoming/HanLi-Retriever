import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import MaskedConv1D, Scale, LayerNorm, Linear

def _conv_stack(
    in_dim,     # channels of the input feature pyramid (2 * embd_dim: video stream + audio stream)
    feat_dim,   # channels inside the head
    n_layers,   # layers of the head, output conv included (so n_layers - 1 are built here)
    ks,         # conv kernel size
    with_ln,    # LayerNorm after every conv
): # not change T
    """Shared body of the heads: n_layers - 1 MaskedConv1D layers, each with its LayerNorm.
    Each head adds its own ReLU and output conv."""
    head, norm = nn.ModuleList(), nn.ModuleList()
    for i in range(n_layers - 1):
        head.append(MaskedConv1D(in_dim if i == 0 else feat_dim, feat_dim, ks,
                                 stride=1, padding=ks // 2, bias=(not with_ln)))
        norm.append(LayerNorm(feat_dim) if with_ln else nn.Identity())
    return head, norm


class EventHead(nn.Module):
    """
    Iterate for each level of T = [256, 128, 64, 32, 16, 8], and check if each level has event occured.

    Returns:
        Tuple[tensor[B, T, 1]]
    """

    def __init__(
        self,
        in_dim,           # channels of the input feature pyramid
        feat_dim,         # channels inside the head
        n_layers=3,       # conv layers, output conv included
        ks=3,             # kernel size
        with_ln=True,     # LayerNorm after every conv
        prior_prob=0.01,  # probability of "event" at init: output bias = -log((1 - p) / p)
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.out = MaskedConv1D(feat_dim, 1, ks, stride=1, padding=ks // 2)
        torch.nn.init.constant_(self.out.conv.bias, -np.log((1 - prior_prob) / prior_prob))

    def forward(
        self,
        fpn_feats,  # 6 levels, each (B, C, T_l)
        fpn_masks,  # 6 levels, each (B, 1, T_l) bool, True = real step (not padding)
    ):
        res = tuple()
        for x, m in zip(fpn_feats, fpn_masks):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            o, _ = self.out(x, m)
            res += (o.permute(0, 2, 1),)          # (B, T_l, 1)
        return res # tuple[tensor(B, T_l, 1)]


class EmbedHead(nn.Module):
    """One L2-normalised vector per time step, in the same space as the projected captions.
    With omi_feats_dim > 0 a second projection puts every step in the OmniRetriever caption space too."""

    def __init__(
        self,
        in_dim,         # channels of the input feature pyramid
        feat_dim,       # channels inside the head, also the size D of every step vector
        n_layers=3,     # conv layers
        ks=3,           # kernel size
        with_ln=True,   # LayerNorm after every conv
        clip_dim=512,   # size of the caption vectors (InternVideo2 text)
        omi_feats_dim=3584
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.vis_proj = MaskedConv1D(feat_dim, feat_dim, ks, stride=1, padding=ks // 2)
        self.omni_vis_proj = None
        if omi_feats_dim > 0:
            self.omni_vis_proj = MaskedConv1D(feat_dim, omi_feats_dim, ks, stride=1, padding=ks // 2)
        self.clip_proj = Linear(clip_dim, feat_dim) # 512 -> 512 : captions
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        fpn_feats,  # 6 levels, each (B, C, T_l)
        fpn_masks,  # 6 levels, each (B, 1, T_l) bool
    ):
        """Returns the normalised vectors of every step (and of the omni projection, empty without it),
        and the raw features of level 0 (the finest) for the span averages."""
        res, raw0 = tuple(), None
        omni_res = tuple()
        for x, m in zip(fpn_feats, fpn_masks):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            x1, _ = self.vis_proj(x, m)
            if raw0 is None:
                raw0 = x1                                       # (B, D, T0)
            res += (F.normalize(x1, dim=1).permute(0, 2, 1),)   # (B, T_l, D)
            
            if self.omni_vis_proj is not None:
                x2, _ = self.omni_vis_proj(x, m)
                omni_res += (F.normalize(x2, dim=1).permute(0, 2, 1),)
                
        return res, omni_res, raw0

    def embed_captions(
        self,
        cap,  # (N, 512) vector caption
    ):
        return F.normalize(self.clip_proj(cap), dim=-1) # (N, 512)


# raw (512, 256) level 0, not normalized
def span_mean(
    raw,     # (D, T) level-0 features of one video, not normalised
    length,  # real steps (padding excluded) of the video
    segs,    # (N, 2) segments (start, end), in level-0 steps
):
    """Mean of the features over every segment.

    Uses a cumulative sum, so there is no loop over the segments. A segment shorter than one step
    still takes at least one step.
    """
    # pad 1 element on left and 0 element on right
    cs = F.pad(raw.cumsum(-1), (1, 0), value=0)                         # (D, T+1)
    lo = segs[:, 0].floor().clamp(min=0, max=length - 1).long() # (N,)
    hi = segs[:, 1].ceil().long() + 1
    hi = torch.maximum(hi.clamp(max=length), lo + 1) # (N,)
    return (cs[:, hi] - cs[:, lo]).t() / (hi - lo).unsqueeze(1).float()   # (N, D)



class SegmentContext(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim + 2, dim), # span + glob + pos = 512 + 512 + 2
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        raw,     # (D, T) level-0 features of one video
        length,  # real steps of the video
        segs,    # (N, 2) segments, in steps
    ):
        span = span_mean(raw, length, segs)                          # (N, D)
        glob = raw[:, :length].mean(-1).expand_as(span)              # (N, D)
        pos = (segs / float(length)).clamp(0, 1)                     # (N, 2)
        return F.normalize(span + self.mlp(torch.cat((span, glob, pos), -1)), dim=-1) # (N, D)


class ArticleCondition(nn.Module):
    def __init__(self, dim, text_dim=512, n_head=8, max_steps=64):
        super().__init__()
        self.txt = nn.Sequential(
            nn.Linear(text_dim, dim), # 512 -> 1024
            nn.LayerNorm(dim)
        )
        self.order = nn.Embedding(max_steps, dim)    # position of the step in the article
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
        
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(
        self,
        feats,     # per level (B, D, T_l)
        masks,     # per level (B, 1, T_l)
        txt,       # (B, C, 512) step texts of every article, padded
        txt_pad,   # (B, C) True on padding
    ):
        C = txt.shape[1]
        kv = self.txt(txt) + self.order(
            torch.arange(C, device=txt.device).clamp(max=self.order.num_embeddings - 1)
        )
        out = []
        for f, m in zip(feats, masks):
            q = self.norm(f.transpose(1, 2))                                  # (B, T_l, D)
            a, _ = self.attn(q, kv, kv, key_padding_mask=txt_pad, need_weights=False) # (, T_l, D)
            out.append(f + a.transpose(1, 2) * m.to(f.dtype))
        return out


class ArticleHead(nn.Module):
    """Picks the video's article (recipe) by its text, so the article needs not be given at inference.
    logits = scale * pooled video @ article vectors; an article vector is the mean of its step text vectors."""

    def __init__(self, dim, head_dim, text_dim=512):
        super().__init__()
        self.vid = nn.Sequential(
            nn.Linear(2 * dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, head_dim)
        )
        self.txt = nn.Linear(text_dim, head_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        f,        # (B, D, T) level-0 features
        m,        # (B, 1, T) bool
        art_txt,  # (A, text_dim) article vectors
    ):
        m = m.to(f.dtype)
        mean = (f * m).sum(-1) / m.sum(-1).clamp(min=1)
        mx = f.masked_fill(m == 0, -1e4).max(-1).values
        v = F.normalize(self.vid(torch.cat((mean, mx), -1)), dim=-1)      # (B, head_dim), head_dim = 2D
        
        t = F.normalize(self.txt(art_txt), dim=-1)                       # (A, head_dim)
        
        return self.logit_scale.exp().clamp(max=100) * v @ t.t()         # (B, A)


class BoundaryHead(nn.Module):
    """At every step: distances to the start and end of the event holding it (in strides of the level,
    always >= 0), and an IoU-quality logit (how good the segment predicted from this step is, for ranking)."""

    def __init__(
        self,
        in_dim,          # channels of the input feature pyramid
        feat_dim,        # channels inside the head
        fpn_levels=6,    # pyramid levels: each level has its own scale for the distances
        n_layers=3,      # conv layers
        ks=3,            # kernel size
        with_ln=True,    # LayerNorm after every conv
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.head, self.norm = _conv_stack(in_dim, feat_dim, n_layers, ks, with_ln)
        self.scale = nn.ModuleList([Scale() for _ in range(fpn_levels)])
        self.out = MaskedConv1D(feat_dim, 2, ks, stride=1, padding=ks // 2)
        torch.nn.init.constant_(self.out.conv.bias, 1.0)
        # predicted IoU between the segment from this step and the GT segment, for ranking
        self.iou_out = MaskedConv1D(feat_dim, 1, ks, stride=1, padding=ks // 2)

    def forward(
        self,
        fpn_feats,  # 6 levels, each (B, C, T_l)
        fpn_masks,  # 6 levels, each (B, 1, T_l) bool
    ):
        res, qual = tuple(), tuple()
        for l, (x, m) in enumerate(zip(fpn_feats, fpn_masks)):
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            q, _ = self.iou_out(x, m)
            qual += (q.permute(0, 2, 1),)                         # (B, T_l, 1)
            o, _ = self.out(x, m)
            res += (F.relu(self.scale[l](o)).permute(0, 2, 1),)  # (B, T_l, 2)
        return res, qual