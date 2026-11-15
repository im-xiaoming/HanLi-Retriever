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
        # skip connection: the head input (pyramid features), projected to the omni space by a 1x1 conv, is added
        # to the omni output, so the omni vector also sees the features before the conv stack
        self.omni_skip = MaskedConv1D(in_dim, omi_feats_dim, 1, stride=1, padding=0)
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
            x_in = x                                            # (B, C, T_l) head input, for the omni skip
            for i in range(len(self.head)):
                x, _ = self.head[i](x, m)
                x = self.act(self.norm[i](x))
            x1, _ = self.vis_proj(x, m)
            if raw0 is None:
                raw0 = x1                                       # (B, D, T0)
            res += (F.normalize(x1, dim=1).permute(0, 2, 1),)   # (B, T_l, D)
            x2, _ = self.omni_vis_proj(x, m)
            x2 = x2 + self.omni_skip(x_in, m)[0]                # (B, 3584, T_l) + skip from the head input
            omni_res += (F.normalize(x2, dim=1).permute(0, 2, 1),)   # (B, T_l, 3584)

        return res, omni_res, raw0

    def embed_captions(
        self,
        cap,  # (N, 512) vector caption
    ):
        return F.normalize(self.clip_proj(cap), dim=-1) # (N, 512)


def pad_segments(
    segs,   # [B] (N_b, 2) segments of every video (N_b may be 0)
):
    """Segments of every video in one tensor, for the batched span heads (one call for the whole batch instead
    of one per video). Returns (B, N, 2) segments, padded with [0, 0], and (B, N) bool, True = a real segment."""
    n = max(max((len(s) for s in segs), default=0), 1)
    out = segs[0].new_zeros(len(segs), n, 2)
    valid = torch.zeros(len(segs), n, dtype=torch.bool, device=segs[0].device)
    for b, s in enumerate(segs):
        out[b, :len(s)] = s
        valid[b, :len(s)] = True
    return out, valid


def segment_masks(
    length,   # (B,) real steps of every video
    segs,     # (B, N, 2) segments (start, end), in steps
    T,        # steps of the padded videos
):
    """Apply mask for each segments."""
    length = length.long()[:, None]                                         # (B, 1)
    lo = torch.minimum(segs[..., 0].floor().long().clamp(min=0), length - 1)  # (B, N) first step
    hi = torch.minimum(segs[..., 1].ceil().long() + 1, length)
    hi = torch.maximum(hi, lo + 1)                                          # (B, N) one past the last step
    t = torch.arange(T, device=segs.device)                                 # (T,)
    return (t >= lo[..., None]) & (t < hi[..., None])                       # (B, N, T)


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
        x,     # (B, T, dim) B videos of T vectors
        mask,  # (B, N, T) bool: N sets per video, True = this vector belongs to the set
    ):
        """(B, N, out_dim): one vector per set. The sets of a video share its vectors, so the scores are computed
        once per video and only the mask differs between its sets. A set with no vector (padding) gets the
        mean of the whole video, to stay finite; it is meant to be dropped by the caller."""
        B, N, T = mask.shape
        dim = x.shape[-1]
        keys = x                                                      # (B, T, dim)
        values = x                                                    # (B, T, dim)
        mask = mask | ~mask.any(-1, keepdim=True)                     # empty (padding) sets: every vector

        # score of query q for input t: dot(query q, key t) / sqrt(dim), as in standard attention
        #                 (Q, dim)       (B, dim, T)
        scores = self.queries @ keys.transpose(1, 2) / dim ** 0.5     # (B, Q, T)
        scores = scores[:, None].masked_fill(~mask[:, :, None], float('-inf'))   # (B, N, Q, T) outside the set: 0
        weights = scores.softmax(dim=-1)                              # (B, N, Q, T), sums to 1 over T

        # every query: weighted mean of the inputs
        #          (B, N * Q, T) @ (B, T, dim)
        pooled = weights.flatten(1, 2) @ values                       # (B, N * Q, dim)
        return self.out(pooled.reshape(B, N, len(self.queries) * dim))   # (B, N, Q * dim) -> (B, N, out_dim)


class OmniSpanHead(nn.Module):
    """One OmniRetriever vector for a whole segment, from every pyramid level.

    A segment [start, end] (level-0 steps) covers [start / s_l, end / s_l] at level l (stride s_l): its steps of
    the 6 levels are put in one set (fine steps for the details, coarse steps for the context), each tagged with a
    learned embedding of its level, and AttentionPool compresses the set into one vector.
    """

    def __init__(
        self,
        dim,            # channels of the pyramid features (1024)
        out_dim=3584,   # size of the OmniRetriever vectors
        n_queries=8,    # learned queries of the attention pooling
        hidden=4096,    # hidden size of the MLP
        strides=(1,),   # stride of every pyramid level, in level-0 steps
    ):
        super().__init__()
        self.strides = list(strides)
        self.level_emb = nn.Parameter(torch.zeros(len(self.strides), dim))   # (L, dim), Xavier init (init_heads)
        self.pool = AttentionPool(dim, out_dim, n_queries, hidden)
        # skip connection: the mean of the segment's steps (all levels), projected linearly to the omni space, is
        # added to the AttentionPool output, a direct path from the features to the omni vector
        self.skip = nn.Linear(dim, out_dim)

    def forward(
        self,
        feats,    # per level (B, D, T_l) pyramid features
        lens,     # (B, L) real steps of every video at every level
        segs,     # (B, N, 2) segments (start, end) in level-0 steps (pad_segments)
    ):
        xs, masks = [], []
        for l, (f, s) in enumerate(zip(feats, self.strides)):
            xs.append(f.transpose(1, 2) + self.level_emb[l])                      # (B, T_l, D)
            masks.append(segment_masks(lens[:, l], segs / s, f.shape[2]))        # (B, N, T_l)
        x = torch.cat(xs, dim=1)                                                  # (B, P, D) every level
        mask = torch.cat(masks, dim=2)                                            # (B, N, P)
        mf = mask.to(x.dtype)
        mean = (mf @ x) / mf.sum(-1, keepdim=True).clamp(min=1)                   # (B, N, D) mean of the segment
        return F.normalize(self.pool(x, mask) + self.skip(mean), dim=-1)          # (B, N, out_dim)


class SegmentContext(nn.Module):
    """Group T vector in the same step into 1 vector. Return (B, N, D) which N is number of segments.
    """
    def __init__(
        self,
        dim: int = 512,
        n_queries: int = 4,   # learned queries of the AttentionPool
        hidden: int = 2048,   # hidden size of the AttentionPool MLP
    ):
        super().__init__()
        self.pool = AttentionPool(dim, dim, n_queries, hidden)   # vector of the span (instead of its mean)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim + 2, dim), # span + glob + pos = 512 + 512 + 2
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(
        self,
        raw,     # (B, D, T) level-0 features
        length,  # (B,) real steps of every video
        segs,    # (B, N, 2) segments, in steps (pad_segments)
    ):
        T = raw.shape[2]
        masks = segment_masks(length, segs, T)                       # (B, N, T)
        span = self.pool(raw.transpose(1, 2), masks)                 # (B, N, D)
        real = (torch.arange(T, device=raw.device) < length[:, None]).to(raw.dtype)   # (B, T) real steps
        glob = (raw * real[:, None]).sum(-1) / length[:, None].to(raw.dtype)          # (B, D) mean of the video
        glob = glob[:, None].expand_as(span)                         # (B, N, D)
        pos = (segs / length[:, None, None].to(segs.dtype)).clamp(0, 1)                # (B, N, 2)
        return F.normalize(span + self.mlp(torch.cat((span, glob, pos), -1)), dim=-1) # (B, N, D)


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
        dim,               # channels of the level-0 features (1024)
        head_dim,          # size of the video and article vectors (512)
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
        # learned part of every article vector (Xavier init, init_heads); with ~9 train videos per article it can
        # learn what the text does not tell apart (similar dishes)
        self.article_learned = nn.Parameter(torch.zeros(num_articles, head_dim))

    def pool(
        self,
        f,  # (B, D, T) features of one pyramid level
        m,  # (B, 1, T) bool, True = real step
    ):
        """Mean and max over the real steps of one level, concatenated: (B, 2D)."""
        m = m.to(f.dtype)
        mean = (f * m).sum(-1) / m.sum(-1).clamp(min=1) # (B, D)
        mx = f.masked_fill(m == 0, -1e4).max(-1).values # (B, D)
        return torch.cat((mean, mx), -1) # (B, 2D)

    def forward(
        self,
        f,        # (B, D, T) level-0 features
        m,        # (B, 1, T) bool, True = real step
        art_txt,  # (A, text_dim) article vectors
    ):
        v = F.normalize(self.vid(self.pool(f, m)), dim=-1)              # (B, head_dim) mean + max, then MLP

        t = F.normalize(self.txt(art_txt) + self.article_learned, dim=-1)   # (A, head_dim)

        return self.logit_scale.exp().clamp(max=100) * v @ t.t()         # (B, A)


def init_heads(
    heads,   # the head modules (not the backbone)
):
    """Xavier-uniform init of every weight of the heads: Linear / conv weights, attention projections and the
    learned vectors (article_learned, level_emb, queries), including the outputs that used to start at zero
    (SegmentContext, ArticleCondition). Biases become 0, except the event prior and the boundary bias (they set
    the starting outputs of the losses). LayerNorm and logit_scale are kept."""
    for head in heads:
        keep_bias = {id(head.out.conv.bias)} if isinstance(head, (EventHead, BoundaryHead)) else set()
        for name, p in head.named_parameters():
            if 'norm' in name or 'logit_scale' in name or id(p) in keep_bias:
                continue                                    # LayerNorm, temperature, prior biases
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)                  # weights (conv: fan over in_channels * kernel)
            elif name.endswith('bias'):
                nn.init.zeros_(p)


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