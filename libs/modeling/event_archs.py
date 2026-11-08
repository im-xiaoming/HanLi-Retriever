import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import register_multimodal_meta_arch, make_multimodal_backbone
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss, info_NCE_loss, omni_loss, omni_span_loss
from ..utils import batched_nms
from .heads import (
    EventHead, BoundaryHead, EmbedHead, SegmentContext, ArticleCondition, ArticleHead, span_mean,
    OmniSpanHead
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
        omni_head_dim,       # size of the OmniRetriever caption vectors (3584), 0 = no omni head
        regression_range,   # each level handles the segments whose largest distance to a boundary falls in this range
        head_num_layers,     # conv layers of every head
        head_kernel_size,    # kernel size of the heads
        head_with_ln,        # LayerNorm in the heads
        use_abs_pe,          # add a sinusoid position embedding in the backbone
        train_cfg,           # loss weights, dropout, droppath, cls_prior_prob
        test_cfg,            # inference settings: thresholds, maximum segments, NMS
        max_seq_len,         # steps of every video (256)
        nmax=48,             # at most this many GT segments per video
        clip_dim=512,        # size of the step text vectors,
        article_cond=False,  # condition the features on the article's step texts (ActionFormer-T)
        article_head=False,  # predict the article from the video; test_cfg['pred_article'] uses it at inference
        article_learned=0,   # article_head only: > 0 = number of articles, each gets a learned vector too
        omni_span_head=False,  # omni: a learned head compresses a (start, end) segment into one omni vector
        pool_queries=4,        # omni span head: learned queries of its attention pooling
        omni_span_hidden=2048, # hidden size of the omni span head MLP
    ):
        super().__init__()
        self.fpn_strides = [scale_factor ** i for i in range(backbone_arch[-1] + 1)] # [1, 2, 4, 8, 16, 32]
        self.max_seq_len = max_seq_len
        self.nmax = nmax
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # conv + transformer backbone, one stream per modality (TemporalMaxer and DyFADet backbones were tried
        # and removed: 24.51 and 24.94 mAP vs 25.26, see docs/experiments_report.md)
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
        self.article_cond = ArticleCondition(D, clip_dim) if article_cond else None
        self.article_head = None
        if article_head:
            self.article_head = ArticleHead(D, head_dim, clip_dim, article_learned)

        # omni head: every point inside a GT segment learns the OmniRetriever vector of that segment (teacher).
        self.use_omni = omni_head_dim > 0
        # omni span head: the omni vector of a GT segment (start, end given) by learned attention pooling.
        self.omni_span = None
        if self.use_omni and omni_span_head:
            self.omni_span = OmniSpanHead(D, omni_head_dim, pool_queries, omni_span_hidden)
        # mean of the train teacher vectors (the part every segment shares), for the OmniCosC metric; set by train.py
        self.omni_teacher_mean = None

    def set_step_pool(
        self,
        pool_raw,  # (4958, clip_dim) InternVideo2 text vectors of the steps, row i = step i
        article_steps=None,  # [617] pool rows of every article's steps (HTStepDataset.articles order)
    ):
        self.pool_raw = pool_raw # pool_emb
        self.article_steps = article_steps
        if article_steps is not None:
            # article vector = mean of its normalised step vectors (A, clip_dim)
            pn = F.normalize(pool_raw, dim=-1)
            self.article_raw = torch.stack([
                #           (n_steps, 512)
                pn[torch.as_tensor(s, device=pn.device)].mean(0) for s in article_steps
            ], dim=0)

            # article_of_step[s] = article of step s (every step belongs to exactly one article)
            self.article_of_step = torch.zeros(len(pool_raw), dtype=torch.long, device=pool_raw.device)
            for article, steps in enumerate(article_steps):
                self.article_of_step[torch.as_tensor(steps, device=pool_raw.device)] = article

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

        # article of every video: given (cand), or predicted by the article head (test_cfg pred_article)
        art_logits = None
        if self.article_head is not None:
            # article_detach: the head reads the features but sends no gradient back into the backbone
            f0 = feats[0].detach() if self.train_cfg.get('article_detach') else feats[0]
            art_logits = self.article_head(f0, msk[0], self.article_raw)  # (B, A)
            
        cands = [x['cand'].to(self.device) for x in video_list]
        if not self.training and self.test_cfg.get('pred_article') and art_logits is not None:
            # steps of the top-k predicted articles, best article first, each step once
            k = self.test_cfg.get('pred_article_topk', 1)
            cands = [torch.as_tensor(list(dict.fromkeys(s for i in top for s in self.article_steps[i])), device=self.device)
                     for top in art_logits.topk(k, -1).indices.tolist()]
            
        # step texts of every video's article, padded: (B, C) pool rows and the padding mask (B, C)
        C = max(len(c) for c in cands)
        idx = torch.stack([F.pad(c, (0, C - len(c)), value=0) for c in cands])
        pad = torch.stack([
            torch.arange(C, device=idx.device) >= len(c) for c in cands
        ], dim=0)
        
        if self.article_cond is not None:
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
        omni_em = torch.cat(omni_em, dim=1) if self.use_omni else None  # (B, P, 3584), normalised

        #       (B, 1, T)
        len0 = msk[0].squeeze(1).sum(-1)                                 # (B,)
        #                         (B, T)
        fpn_masks = torch.cat([m.squeeze(1) for m in msk], dim=1)        # (B, P)
        if self.training:
            return self.losses(video_list, fpn_masks, ev, bd, qu, em, omni_em, raw0, len0, art_logits, feats[0])
        return self.inference(video_list, fpn_masks, ev, bd, qu, raw0, len0, cands, art_logits, omni_em, feats[0])

    def losses(
        self,
        video_list,  # B dicts, one per video
        valid,       # (B, P) bool, True at points of real steps (not padding)
        ev,          # (B, P) event head logits
        bd,          # (B, P, 2) predicted distances to the start / end, in strides
        qu,          # (B, P) IoU-quality logits
        em,          # (B, P, D) normalised embedding of every step
        omni_em,     # (B, P, D)
        raw0,        # (B, D, T) level-0 features of the embedding head, not normalised (for the span means)
        len0,        # (B,) real level-0 steps of every video
        art_logits,  # (B, A) article head logits; None = no article head
        feat0=None,  # (B, D, T) level-0 pyramid features (for the omni span head)
    ):
        dev = ev.device
        w = self.train_cfg
        gt_cls = torch.stack([x['gt_cls_labels'].to(dev) for x in video_list])   # (B, P, 48)
        gt_off = torch.stack([x['gt_offsets'].to(dev) for x in video_list])      # (B, P, 2)
        cap_pool = torch.stack([x['step_idx'] for x in video_list]).to(dev)      # (B, 48) step of every GT segment

        pool_f = self.embed_head.embed_captions(self.pool_raw)          # (N, 512)
        
        is_event = gt_cls.sum(-1) > 0                                    # (B, P)
        pos = is_event & valid # (B, P)
        n_pos = max(int(pos.sum()), 1) #  >= 1

        
        # ========================== Regression Loss ====================================
        #                           (N, 2)     (N, 2)
        rg_loss, iouk = ctr_diou_loss_1d(bd[pos], gt_off[pos], reduction='sum')
        rg_loss = rg_loss / n_pos

        # ============================================= Event Loss ====================================
        #                               (V,)          (V,)
        ev_loss = sigmoid_focal_loss(ev[valid], is_event[valid].float(), reduction='sum') / n_pos

        # ======================================= IoU-quality Loss ===================================
        iou_loss = F.binary_cross_entropy_with_logits(qu[pos], iouk)

        # ========================================== InfoNCE Loss =======================================
        scale = self.embed_head.logit_scale.exp().clamp(max=100)
        
        ri = pos.nonzero(as_tuple=True)[0]
        gt = cap_pool[ri, gt_cls[pos].argmax(-1)] # (N',)
        #                                   (N', 512)    (512, N)
        emb_loss = info_NCE_loss(em[pos], pool_f, gt, scale)
        #                                       cosin matrix
        
        # <<<<<<<<<<<<<<<< Omni >>>>>>>>>>>>>>
        omni_emb_loss = torch.zeros((), device=dev)
        omni_cos_loss = torch.zeros((), device=dev)
        omni_span_nce = torch.zeros((), device=dev)
        omni_span_cos = torch.zeros((), device=dev)
        if self.omni_span is not None:
            span_losses = self.omni_span_losses(video_list, feat0, len0, cap_pool)
            omni_span_nce = span_losses['nce']
            omni_span_cos = span_losses['cos']
        if self.use_omni:
            omni_losses = omni_loss(video_list, pos, gt_cls, cap_pool, omni_em,
                                    self.train_cfg.get('omni_temperature', 0.05))
            omni_emb_loss = omni_losses['nce']
            omni_cos_loss = omni_losses['cos']

        # =========================================== Span Loss =========================================
        qs, gs = [], []
        for i, x in enumerate(video_list):
            #                       (D, T)
            qs.append(self.seg_ctx(raw0[i], int(len0[i]), x['segments'].to(dev).float())) # (M, D)
            gs.append(cap_pool[i, x['labels'].to(dev)]) # (M,)
        #                                   (M, D) x (D, N) = (M, N)
        span_loss = info_NCE_loss(
            torch.cat(qs),
            pool_f,
            torch.cat(gs),
            scale
        )

        # 6. article head: the video must pick its own article among all of them
        art_loss = torch.zeros((), device=dev)
        if art_logits is not None:
            art_loss = F.cross_entropy(art_logits, torch.as_tensor([x['article'] for x in video_list], device=dev))


        total = (ev_loss + w['loss_weight_reg'] * rg_loss + w['loss_weight_iou'] * iou_loss
                 + w['loss_weight_emb'] * emb_loss + w['loss_weight_omni_emb'] * omni_emb_loss
                 + w.get('loss_weight_omni_cos', 0.0) * omni_cos_loss
                 + w.get('loss_weight_omni_span_nce', 0.0) * omni_span_nce
                 + w.get('loss_weight_omni_span_cos', 0.0) * omni_span_cos
                 + w['loss_weight_span'] * span_loss
                 + w.get('loss_weight_article', 0.0) * art_loss)

        return {'ev_loss': ev_loss, 'reg_loss': rg_loss, 'iou_loss': iou_loss, 'emb_loss': emb_loss,
                'omni_emb_loss': omni_emb_loss, 'omni_cos_loss': omni_cos_loss,
                'omni_span_nce': omni_span_nce, 'omni_span_cos': omni_span_cos, 'span_loss': span_loss,
                'art_loss': art_loss, 'final_loss': total}

    def omni_span_losses(
        self,
        video_list,  # B dicts (segments in steps after the crop, labels = slots, omni_seg, omni_ok, seg_key)
        feat0,       # (B, D, T) level-0 pyramid features
        len0,        # (B,) real level-0 steps of every video
        cap_pool,    # (B, 48) step (pool row) of every slot
    ):
        """Omni span head against the teachers: one anchor per GT segment (start and end given)."""
        dev = feat0.device
        students, teachers, steps, keys = [], [], [], []
        for i, x in enumerate(video_list):
            slots = x['labels'].to(dev)                                   # (M,) slot of every kept GT segment
            has_teacher = x['omni_ok'].to(dev)[slots]                     # (M,)
            if not has_teacher.any():
                continue
            slots = slots[has_teacher]
            segments = x['segments'].to(dev).float()[has_teacher]         # (M', 2) in level-0 steps
            students.append(self.omni_span(feat0[i], int(len0[i]), segments))   # (M', 3584)
            teachers.append(x['omni_seg'].to(dev)[slots])
            steps.append(cap_pool[i, slots])
            keys.append(x['seg_key'].to(dev)[slots])
        if not students:
            zero = feat0.sum() * 0.0
            return {'nce': zero, 'cos': zero}
        return omni_span_loss(torch.cat(students), torch.cat(teachers), torch.cat(steps), torch.cat(keys),
                              self.train_cfg.get('omni_temperature', 0.05))

    @torch.no_grad()
    def omni_eval(
        self,
        omni_em_one,  # (P, 3584) student vectors of one video, L2-normalised
        length,       # real level-0 steps of the video
        video,        # the dict of the video (segments in steps, omni_seg, omni_ok)
        feat0_one=None,  # (D, T) level-0 pyramid features of the video (for the omni span head)
    ):
        """Student vs teacher for every GT segment of one validation video that has a teacher vector.
        The GT start and end are given. Student vector of a segment = the omni span head on it if the model has
        one, else the mean of the level-0 student vectors over the segment; L2-normalised."""
        has_teacher = video['omni_ok'][:len(video['segments'])]       # (n,)
        if not has_teacher.any():
            return None, None
        segments = video['segments'][has_teacher].to(omni_em_one.device).float()   # (n', 2) in steps
        if self.omni_span is not None:
            students = self.omni_span(feat0_one, length, segments)                # (n', 3584)
        else:
            level0 = omni_em_one[:self.max_seq_len].t()                # (3584, T0): level 0 = first T0 points
            students = F.normalize(span_mean(level0, length, segments), dim=-1)    # (n', 3584)
        teachers = video['omni_seg'][:len(video['segments'])][has_teacher]     # (n', 3584)
        return students.cpu(), teachers.cpu()

    @torch.no_grad()
    def inference(
        self,
        video_list,  # B dicts, one per video
        valid,       # (B, P) bool, True at points of real steps
        ev,          # (B, P) event head logits
        bd,          # (B, P, 2) predicted distances to the start / end, in strides
        qu,          # (B, P) IoU-quality logits
        raw0,        # (B, D, T) level-0 features of the embedding head (for the segment vectors)
        len0,        # (B,) real level-0 steps of every video
        cands,       # [B] pool rows of every video's article steps (given or predicted)
        art_logits,  # (B, A) article head logits; None = no article head
        omni_em=None,  # (B, P, 3584) student vectors of the omni head; None = no omni head
        feat0=None,    # (B, D, T) level-0 pyramid features (for the omni span head)
    ):
        """Score of a segment = event score times IoU-quality score. Returns one dict per video:
        segments (K, 2) in seconds, best first; labels (K,) the step of every segment, label_probs (K,) its
        probability; gt_segments, gt_text the GT steps (seconds) and their captions.
        """
        t = self.test_cfg
        a = t['iou_power']
        prob = ev.sigmoid().pow(1 - a) * qu.sigmoid().pow(a) * valid.float() # (B, P)
        pool_f = self.embed_head.embed_captions(self.pool_raw)              # (4958, D)
        scale = self.embed_head.logit_scale.exp().clamp(max=100)
        out = []
        for b, v in enumerate(video_list):
            pts = torch.cat(v['points'], dim=0).to(ev.device)   # (P, 4): t, regression range low, high, stride
            #          (P,)
            keep = (prob[b] > t['score_thresh']).nonzero(as_tuple=True)[0] # row
            if keep.numel() == 0:
                keep = prob[b].topk(1).indices
            sc, order = prob[b][keep].topk(min(t['max_seg_num'] * 5, keep.numel()))
            keep = keep[order]
            # segment of a point: from t - (distance to start) * stride to t + (distance to end) * stride
            p, o = pts[keep], bd[b][keep] # (P', 4), (P', 2)
            #                     (P',) - (P',) * (P',) = (P',)
            segs = torch.stack((p[:, 0] - o[:, 0] * p[:, 3], p[:, 0] + o[:, 1] * p[:, 3]), -1).cpu() # (P', 2)
            scores = sc.cpu() # (P',)
            ok = (segs[:, 1] - segs[:, 0]) > t['duration_thresh']
            segs, scores = segs[ok], scores[ok]
            # class-agnostic soft-NMS: every segment shares one dummy label
            segs, scores, _ = batched_nms(segs, scores, torch.arange(len(segs)), 0.7, 0.001,
                                          t['max_seg_num'], use_soft_nms=True, sigma=t['nms_sigma'])
            
            # vector of the segment = mean over the whole segment, computed while it is still in steps
            #                      (D, T0)
            span = self.seg_ctx(raw0[b], int(len0[b]), segs.to(raw0.device)) # (N, D)

            # step of every segment, among the steps of the article (or all of them)
            # cands: list tensor 1D
            if t.get('article_soft') and art_logits is not None:
                # soft article choice: every step of the taxonomy competes, and each one adds the log-probability
                # of its article given the video (article head). A wrong top-1 article no longer forbids the
                # right step: a step that matches the segment well can still win through a likely-enough article.
                cand = torch.arange(len(pool_f), device=ev.device)                   # (4958,)
                log_p_step = F.log_softmax(scale * span @ pool_f.t(), dim=-1)        # (K, 4958)
                log_p_article = F.log_softmax(art_logits[b].float(), dim=-1)         # (A,)
                logp = log_p_step + t.get('article_soft_weight', 1.0) * log_p_article[self.article_of_step]
            else:
                cand = cands[b] if t['article'] else torch.arange(len(pool_f), device=ev.device)
                logp = F.log_softmax(scale * span @ pool_f[cand].t(), dim=-1)    # (K, C)
            lp, j = logp.max(-1) # j = (K,)

            # steps to seconds (see __getitem__ of the dataset)
            st, nf, fps = v['feat_stride'], v['feat_num_frames'], v['fps']
            segs = ((segs * st + 0.5 * nf) / fps).clamp(min=0.0, max=float(v['duration']))

            # omni head: student and teacher vector of every GT segment, for the distance between them
            omni_student, omni_teacher = None, None
            if omni_em is not None:
                omni_student, omni_teacher = self.omni_eval(omni_em[b], int(len0[b]), v, feat0[b])

            out.append({'video_id': v['video_id'], 'segments': segs, 'scores': scores,
                        'labels': cand[j].cpu(), 'label_probs': lp.exp().cpu(),
                        'article_pred': int(art_logits[b].argmax()) if art_logits is not None else -1,
                        'omni_student': omni_student, 'omni_teacher': omni_teacher,
                        'gt_segments': v['segments_sec'], 'gt_text': v['cap_text']})
        return out
