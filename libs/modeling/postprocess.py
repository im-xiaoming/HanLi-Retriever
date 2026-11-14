"""Inference of EventCaptionTransformer: from the outputs of its heads to segments in seconds, each with a step."""
import torch
from torch.nn import functional as F

from ..utils import batched_nms


@torch.no_grad()
def omni_gt_vectors(
    omni_span,  # the omni span head of the model
    video,      # the dict of one validation video (segments in steps, omni_seg, omni_ok)
    feat0,      # (D, T) level-0 pyramid features of the video
    length,     # real level-0 steps of the video
):
    """Span head vs teacher on every GT segment of the video that has a teacher (start and end given)."""
    has_teacher = video['omni_ok'][video['labels']]                          # (n,)
    segments = video['segments'][has_teacher].to(feat0.device).float()       # (n', 2) in steps
    students = omni_span(feat0, length, segments)                             # (n', 3584)
    teachers = video['omni_seg'][video['labels']][has_teacher]                # (n', 3584)
    return students.cpu(), teachers.cpu()


@torch.no_grad()
def event_inference(
    model,       # the EventCaptionTransformer (its heads, test_cfg and step pool)
    video_list,  # B dicts, one per video
    valid,       # (B, P) bool, True at points of real steps
    ev,          # (B, P) event head logits
    bd,          # (B, P, 2) predicted distances to the start / end, in strides
    qu,          # (B, P) IoU-quality logits
    raw0,        # (B, D, T) level-0 features of the embedding head (for the segment vectors)
    len0,        # (B,) real level-0 steps of every video
    cands,       # [B] pool rows of every video's article steps (given)
    art_logits,  # (B, A) article head logits
    feat0,       # (B, D, T) level-0 pyramid features (for the omni span head)
):
    """Score of a segment = event score times IoU-quality score. Returns one dict per video:
    segments (K, 2) in seconds, best first; labels (K,) the step of every segment, label_probs (K,) its
    probability; gt_segments, gt_text the GT steps (seconds) and their captions.
    """
    t = model.test_cfg
    a = t['iou_power']
    prob = ev.sigmoid().pow(1 - a) * qu.sigmoid().pow(a) * valid.float() # (B, P)
    pool_f = model.embed_head.embed_captions(model.pool_raw)              # (4958, D)
    scale = model.embed_head.logit_scale.exp().clamp(max=100)
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
        span = model.seg_ctx(raw0[b], int(len0[b]), segs.to(raw0.device)) # (N, D)

        # step of every segment, among the steps of the article
        # cands: list tensor 1D
        if t['article_soft']:
            # soft article choice: every step of the taxonomy competes, and each one adds the log-probability
            # of its article given the video (article head). A wrong top-1 article no longer forbids the
            # right step: a step that matches the segment well can still win through a likely-enough article.
            cand = torch.arange(len(pool_f), device=ev.device)                   # (4958,)
            log_p_step = F.log_softmax(scale * span @ pool_f.t(), dim=-1)        # (K, 4958)
            log_p_article = F.log_softmax(art_logits[b].float(), dim=-1)         # (A,)
            logp = log_p_step + t['article_soft_weight'] * log_p_article[model.article_of_step]
        else:
            cand = cands[b]
            logp = F.log_softmax(scale * span @ pool_f[cand].t(), dim=-1)    # (K, C)
        lp, j = logp.max(-1) # j = (K,)

        # steps to seconds (see __getitem__ of the dataset)
        st, nf, fps = v['feat_stride'], v['feat_num_frames'], v['fps']
        segs = ((segs * st + 0.5 * nf) / fps).clamp(min=0.0, max=float(v['duration']))

        # omni span head: student and teacher vector of every GT segment, for the distance between them
        omni_student, omni_teacher = omni_gt_vectors(model.omni_span, v, feat0[b], int(len0[b]))

        out.append({'video_id': v['video_id'], 'segments': segs, 'scores': scores,
                    'labels': cand[j].cpu(), 'label_probs': lp.exp().cpu(),
                    'article_pred': int(art_logits[b].argmax()),
                    'omni_student': omni_student, 'omni_teacher': omni_teacher,
                    'gt_segments': v['segments_sec'], 'gt_text': v['cap_text']})
    return out
