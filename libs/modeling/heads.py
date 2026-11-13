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
    A second projection puts every step in the OmniRetriever space too (the omni point head)."""

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
        self.omni_vis_proj = MaskedConv1D(feat_dim, omi_feats_dim, ks, stride=1, padding=ks // 2)
        self.clip_proj = Linear(clip_dim, feat_dim) # 512 -> 512 : captions
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        fpn_feats,  # 6 levels, each (B, C, T_l)
        fpn_masks,  # 6 levels, each (B, 1, T_l) bool
    ):
        """Returns the normalised vectors of every step, the normalised omni vectors of every step,
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
            x2, _ = self.omni_vis_proj(x, m)
            omni_res += (F.normalize(x2, dim=1).permute(0, 2, 1),)   # (B, T_l, 3584)

        return res, omni_res, raw0

    def embed_captions(
        self,
        cap,  # (N, 512) vector caption
    ):
        return F.normalize(self.clip_proj(cap), dim=-1) # (N, 512)


def segment_masks(
    length,   # real level-0 steps of the video
    segs,     # (N, 2) segments (start, end), in level-0 steps
    T,        # level-0 steps of the padded video
):
    """(N, T) bool: the level-0 steps covered by every segment, from floor(start) to ceil(end) included,
    inside the real steps, at least one step."""
    lo = segs[:, 0].floor().clamp(min=0, max=length - 1).long()             # (N,) first step
    hi = segs[:, 1].ceil().long() + 1
    hi = torch.maximum(hi.clamp(max=length), lo + 1)                        # (N,) one past the last step
    t = torch.arange(T, device=segs.device)                                 # (T,)
    #           (1, T)      (N, 1)         (1, T)       (N, 1)
    return (t[None, :] >= lo[:, None]) & (t[None, :] < hi[:, None]) # (N, T)


# raw (512, 256) level 0, not normalized
def span_mean(
    raw,     # (D, T) level-0 features of one video, not normalised
    length,  # real steps (padding excluded) of the video
    segs,    # (N, 2) segments (start, end), in level-0 steps
):
    """Mean of the features over every segment: (N, D)."""
    masks = segment_masks(length, segs, raw.shape[1]).to(raw.dtype)         # (N, T)
    #         (N, T) @ (T, D)
    return (masks @ raw.t()) / masks.sum(dim=1, keepdim=True)               # (N, D)


class AttentionPool(nn.Module):
    """Learned compression of a set of vectors into one vector (instead of a plain mean).

    Attention with learned queries: Q query vectors (parameters) attend to the T input vectors of the set
    (keys = values = the inputs). Every query gives a weighted mean of the inputs, so it can focus on the vectors
    that matter (e.g. the frames where the action is visible). The Q results are concatenated and an MLP maps them
    to the output size.
    """

    def __init__(
        self,
        dim,            # size of every input vector
        out_dim,        # size of the output vector
        n_queries=4,    # learned queries (each one looks for something different)
        hidden=2048,    # hidden size of the output MLP
    ):
        super().__init__()
        # small at the start: every score is ~0, so every query starts as the plain mean of the set
        self.queries = nn.Parameter(0.02 * torch.randn(n_queries, dim))   # (Q, dim)
        self.out = nn.Sequential(
            nn.LayerNorm(n_queries * dim),
            nn.Linear(n_queries * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self,
        x,     # (N, T, dim) N sets of T vectors
        mask,  # (N, T) bool, True = this vector belongs to the set (at least one True per row)
    ):
        dim = x.shape[-1]
        keys = x                                                      # (N, T, dim)
        values = x                                                    # (N, T, dim)

        # score of query q for input t: dot(query q, key t) / sqrt(dim), as in standard attention
        #                 (Q, dim)       (N, dim, T)
        scores = self.queries @ keys.transpose(1, 2) / dim ** 0.5     # (N, Q, T)
        scores = scores.masked_fill(~mask[:, None, :], float('-inf')) # inputs outside the set get weight 0
        weights = scores.softmax(dim=-1)                              # (N, Q, T), sums to 1 over T

        # every query: weighted mean of the inputs
        #          (N, Q, T) @ (N, T, dim)
        pooled = weights @ values                                     # (N, Q, dim)
        return self.out(pooled.flatten(1))                            # (N, Q * dim) -> (N, out_dim)


class OmniSpanHead(nn.Module):
    """Omni vector of a segment given its start and end: learned attention pooling of the level-0 features of
    the segment, then an MLP to the OmniRetriever space (3584), L2-normalised. It is trained against the teacher
    vector of the segment, so the vector it gives for a GT segment is exactly what OmniCos measures."""

    def __init__(
        self,
        dim,            # channels of the level-0 features (1024)
        out_dim=3584,   # size of the OmniRetriever vectors
        n_queries=8,    # learned queries of the attention pooling
        hidden=4096,    # hidden size of the MLP
    ):
        super().__init__()
        self.pool = AttentionPool(dim, out_dim, n_queries, hidden)

    def forward(
        self,
        feat0,    # (D, T) level-0 features of one video
        length,   # real level-0 steps of the video
        segs,     # (N, 2) segments (start, end) in level-0 steps
    ):
        masks = segment_masks(length, segs, feat0.shape[1])            # (N, T)
        x = feat0.t()[None].expand(len(segs), -1, -1)                  # (N, T, D) the same video for every segment
        return F.normalize(self.pool(x, masks), dim=-1)                # (N, out_dim)


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

    def __init__(
        self,
        dim,               # channels of the level-0 features
        head_dim,          # size of the video and article vectors
        text_dim=512,      # size of the step text vectors
        num_articles=617,  # every article also gets its own learned vector, added to its text vector
    ):
        super().__init__()
        self.vid = nn.Sequential(
            nn.Linear(2 * dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, head_dim)
        )
        self.txt = nn.Linear(text_dim, head_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # learned part of every article vector. It starts at 0, so at first an article is its text only; with
        # ~9 train videos per article it can learn what the text does not tell apart (similar dishes)
        self.article_learned = nn.Parameter(torch.zeros(num_articles, head_dim))

    def pool(
        self,
        f,  # (B, D, T) features of one pyramid level
        m,  # (B, 1, T) bool, True = real step
    ):
        """Mean and max over the real steps of one level, concatenated: (B, 2D)."""
        m = m.to(f.dtype)
        mean = (f * m).sum(-1) / m.sum(-1).clamp(min=1)
        mx = f.masked_fill(m == 0, -1e4).max(-1).values
        return torch.cat((mean, mx), -1)

    def forward(
        self,
        f,        # (B, D, T) level-0 features
        m,        # (B, 1, T) bool, True = real step
        art_txt,  # (A, text_dim) article vectors
    ):
        v = F.normalize(self.vid(self.pool(f, m)), dim=-1)              # (B, head_dim) mean + max, then MLP

        t = F.normalize(self.txt(art_txt) + self.article_learned, dim=-1)   # (A, head_dim)

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