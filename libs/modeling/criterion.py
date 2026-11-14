"""Training losses of EventCaptionTransformer, from the outputs of its heads."""
import torch
from torch.nn import functional as F

from .losses import ctr_diou_loss_1d, sigmoid_focal_loss, info_NCE_loss, omni_losses


def event_losses(
    model,       # the EventCaptionTransformer (its heads and train_cfg)
    video_list,  # B dicts, one per video
    valid,       # (B, P) bool, True at points of real steps (not padding)
    ev,          # (B, P) event head logits
    bd,          # (B, P, 2) predicted distances to the start / end, in strides
    qu,          # (B, P) IoU-quality logits
    em,          # (B, P, D) normalised embedding of every step
    omni_em,     # (B, P, D)
    raw0,        # (B, D, T) level-0 features of the embedding head, not normalised (for the span means)
    len0,        # (B,) real level-0 steps of every video
    art_logits,  # (B, A) article head logits
    feat0,       # (B, D, T) level-0 pyramid features (for the omni span head)
):
    dev = ev.device
    w = model.train_cfg
    gt_cls = torch.stack([x['gt_cls_labels'].to(dev) for x in video_list])   # (B, P, 48)
    gt_off = torch.stack([x['gt_offsets'].to(dev) for x in video_list])      # (B, P, 2)
    cap_pool = torch.stack([x['step_idx'] for x in video_list]).to(dev)      # (B, 48) step of every GT segment

    pool_f = model.embed_head.embed_captions(model.pool_raw)          # (N, 512)

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
    scale = model.embed_head.logit_scale.exp().clamp(max=100)

    ri = pos.nonzero(as_tuple=True)[0]
    gt = cap_pool[ri, gt_cls[pos].argmax(-1)] # (N',)
    #                                   (N', 512)    (512, N)
    emb_loss = info_NCE_loss(em[pos], pool_f, gt, scale)
    #                                       cosin matrix

    # <<<<<<<<<<<<<<<< Omni >>>>>>>>>>>>>>
    omni_point, omni_span = omni_losses(model.omni_span, video_list, pos, gt_cls, omni_em, feat0, len0)

    # =========================================== Span Loss =========================================
    qs, gs = [], []
    for i, x in enumerate(video_list):
        #                       (D, T)
        qs.append(model.seg_ctx(raw0[i], int(len0[i]), x['segments'].to(dev).float())) # (M, D)
        gs.append(cap_pool[i, x['labels'].to(dev)]) # (M,)
    #                                   (M, D) x (D, N) = (M, N)
    span_loss = info_NCE_loss(
        torch.cat(qs),
        pool_f,
        torch.cat(gs),
        scale
    )

    # 6. article head: the video must pick its own article among all of them
    art_loss = F.cross_entropy(art_logits, torch.as_tensor([x['article'] for x in video_list], device=dev))


    total = (ev_loss + w['loss_weight_reg'] * rg_loss + w['loss_weight_iou'] * iou_loss
             + w['loss_weight_emb'] * emb_loss
             + w['loss_weight_omni_kd'] * omni_point['kd'] + w['loss_weight_omni_cos'] * omni_point['cos']
             + w['loss_weight_omni_kd'] * omni_span['kd'] + w['loss_weight_omni_cos'] * omni_span['cos']
             + w['loss_weight_span'] * span_loss
             + w['loss_weight_article'] * art_loss)

    return {'ev_loss': ev_loss, 'reg_loss': rg_loss, 'iou_loss': iou_loss, 'emb_loss': emb_loss,
            'omni_point_kd': omni_point['kd'], 'omni_point_cos': omni_point['cos'],
            'omni_span_kd': omni_span['kd'], 'omni_span_cos': omni_span['cos'], 'span_loss': span_loss,
            'art_loss': art_loss, 'final_loss': total}
