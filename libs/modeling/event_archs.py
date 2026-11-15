import torch
from torch import nn
from torch.nn import functional as F

from .models import register_multimodal_meta_arch, make_multimodal_backbone
from .criterion import event_losses
from .losses import TeacherQueue
from .postprocess import event_inference
from .heads import (
    EventHead, BoundaryHead, EmbedHead, SegmentContext, ArticleCondition, ArticleHead, OmniSpanHead, init_heads
)
from ..datasets.helper import preprocessing


@register_multimodal_meta_arch("EventCaptionTransformer")
class EventCaptionTransformer(nn.Module):
    def __init__(
        self,
        backbone_arch,       # (#conv, #transformer in the stem, #transformer in the pyramid branch), e.g. (2, 3, 5)
        scale_factor,        # length ratio between two pyramid levels
        input_dim_V,         # video feature channels (768)
        input_dim_A,         # audio feature channels (768)
        n_head,              # attention heads in the backbone
        embd_kernel_size,    # kernel size of the embedding convs in the backbone
        embd_dim,            # channels per backbone stream (512); the heads read both streams: 2 * embd_dim
        embd_with_ln,        # LayerNorm after the embedding convs
        head_dim,            # channels in every head, also the size D of the step / segment vectors
        omni_head_dim,       # size of the OmniRetriever vectors (3584)
        regression_range,   # each level handles the segments whose largest distance to a boundary falls in this range
        head_num_layers,     # conv layers of every head
        head_kernel_size,    # kernel size of the heads
        head_with_ln,        # LayerNorm in the heads
        use_abs_pe,          # add a sinusoid position embedding in the backbone
        train_cfg,           # loss weights, dropout, droppath, cls_prior_prob
        test_cfg,            # inference settings: thresholds, maximum segments, NMS
        max_seq_len,         # steps of every video (256)
        num_articles,        # articles (recipes) of the taxonomy (617), each gets a learned vector in the article head
        clip_dim=512,        # size of the step text vectors,
        pool_queries=8,      # omni span head: learned queries of its attention pooling
        omni_span_hidden=4096,  # hidden size of the omni span head MLP
    ):
        super().__init__()
        self.fpn_strides = [scale_factor ** i for i in range(backbone_arch[-1] + 1)] # [1, 2, 4, 8, 16, 32]
        self.max_seq_len = max_seq_len
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.backbone = make_multimodal_backbone(
            'convTransformer',
            **{
                'n_in_V': input_dim_V, 'n_in_A': input_dim_A, 'n_embd': embd_dim, 'n_head': n_head,
                'n_embd_ks': embd_kernel_size, 'max_len': max_seq_len, 'arch': backbone_arch,
                'scale_factor': scale_factor, 'with_ln': embd_with_ln, 'attn_pdrop': 0.0,
                'proj_pdrop': train_cfg['dropout'], 'path_pdrop': train_cfg['droppath'], 'use_abs_pe': use_abs_pe,
            }
        )
        D = embd_dim * 2 # 512 * 2
        self.event_head = EventHead(D, head_dim, head_num_layers, head_kernel_size,
                                    head_with_ln, train_cfg['cls_prior_prob'])
        self.bound_head = BoundaryHead(D, head_dim, len(self.fpn_strides),
                                       head_num_layers, head_kernel_size, head_with_ln)
        self.embed_head = EmbedHead(D, head_dim, head_num_layers, head_kernel_size,
                                    head_with_ln, clip_dim, omni_head_dim)
        self.seg_ctx = SegmentContext(head_dim)
        # condition the features on the article's step texts (ActionFormer-T of the HT-Step paper)
        self.article_cond = ArticleCondition(D, clip_dim)
        # predict the video's article from the video, so it needs not be given at inference
        self.article_head = ArticleHead(D, head_dim, clip_dim, num_articles)

        # omni: the model learns the OmniRetriever vector (teacher) of every GT segment, twice:
        #   - point head (in embed_head): every point inside a GT segment predicts the vector of its segment
        #   - span head: given the start and end of a segment, one vector for the whole segment
        self.omni_span = OmniSpanHead(D, omni_head_dim, pool_queries, omni_span_hidden, self.fpn_strides)
        init_heads([self.event_head, self.bound_head, self.embed_head, self.seg_ctx, self.article_cond,
                    self.article_head, self.omni_span])   # Xavier init of every head
        # mean of the train teacher vectors (the part every segment shares), for the OmniCosC metric; set by train.py
        self.omni_teacher_mean = None
        # teachers of the last batches, extra negatives of the omni InfoNCE (not saved in the checkpoints)
        self.omni_queue = TeacherQueue(train_cfg['omni_queue_size'])

    def set_step_pool(
        self,
        pool_raw,  # (N, D) InternVideo2 text vectors of the steps, row i = step i
        article_steps,  # [617] pool rows of every article's steps (HTStepDataset.articles order)
    ):
        self.pool_raw = pool_raw # pool_emb
        self.article_steps = article_steps
        # article vector = mean of its normalised step vectors (A, clip_dim)
        pn = F.normalize(pool_raw, dim=-1)
        self.article_raw = torch.stack([
            pn[torch.as_tensor(s, device=pn.device)].mean(0) for s in article_steps
        ], dim=0)

        # article_of_step[s] = article of step s (every step belongs to exactly one article)
        self.article_of_step = torch.zeros(len(pool_raw), dtype=torch.long, device=pool_raw.device) # (N,)
        for i, steps in enumerate(article_steps):
            self.article_of_step[torch.as_tensor(steps, device=pool_raw.device)] = i

    @property
    def device(self):
        return list(set(p.device for p in self.parameters()))[0]


    def forward(self, video_list):
        V, A, masks = preprocessing(
            video_list=video_list,
            padding_val=0,
            max_seq_len=self.max_seq_len,
            device=self.device
        ) # V = A = (B, C, T)

        # RETURN PYRAMID FEATURE
        #     tuple[(B, 1, T]
        fV, fA, msk = self.backbone(V, A, masks) # fV = fA = (B, C, T_l), l ∈ [0, 5]

        feats = [torch.cat((v, a), dim=1) for v, a in zip(fV, fA)]           # per level (B, 1024, T_l), len(feats) = 6

        # article head: score of every article for every video, from the features before the article conditioning
        art_logits = self.article_head(feats[0], msk[0], self.article_raw)  # (B, A)

        # steps of the video's article (given)
        cands = [x['cand'].to(self.device) for x in video_list]
        # step texts of every video's article, padded: (B, C) pool rows and the padding mask (B, C)
        C = max(len(c) for c in cands)
        idx = torch.stack([F.pad(c, (0, C - len(c)), value=0) for c in cands])
        pad = torch.stack([
            torch.arange(C, device=idx.device) >= len(c) for c in cands
        ], dim=0)
        
        #                                           (B, C, 512)
        feats = self.article_cond(feats, msk, self.pool_raw[idx], pad)

        # Event Head
        #                   tuple(tensor(B, T_l, 1))
        ev = torch.cat(self.event_head(feats, msk), dim=1).squeeze(-1)   # (B, P), P = point = 504

        # Bounding Head
        bd, qu = self.bound_head(feats, msk)
        bd = torch.cat(bd, dim=1)                                        # (B, P, 2)
        qu = torch.cat(qu, dim=1).squeeze(-1)                            # (B, P)

        # Embedding Head
        em, omni_em, raw0 = self.embed_head(feats, msk) # tuple((B, T_l, D)), (B, D, T0)
        em = torch.cat(em, dim=1)                                        # (B, P, D)
        omni_em = torch.cat(omni_em, dim=1)                              # (B, P, 3584), normalised

        #       (B, 1, T)
        len0 = msk[0].squeeze(1).sum(-1)                                 # (B,)
        lens = torch.stack([m.squeeze(1).sum(-1) for m in msk], dim=1)    # (B, L) real steps of every level
        #                         (B, T)
        fpn_masks = torch.cat([m.squeeze(1) for m in msk], dim=1)        # (B, P)
        if self.training:
            return event_losses(self, video_list, fpn_masks, ev, bd, qu, em, omni_em, raw0, len0, art_logits,
                                feats, lens)
        return event_inference(self, video_list, fpn_masks, ev, bd, qu, raw0, len0, cands, art_logits, feats, lens)
