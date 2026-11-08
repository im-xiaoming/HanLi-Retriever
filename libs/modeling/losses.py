import torch
from torch.nn import functional as F
from torch import Tensor


def info_NCE_loss(logits, C, targets, scale):
    return F.cross_entropy(scale * logits @  C.t(), targets)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Focal loss of RetinaNet (https://arxiv.org/abs/1708.02002), taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    inputs: logits, any shape
    targets: 0 / 1 labels, same shape as inputs
    alpha: weight balancing positive and negative samples (default 0.25)
    gamma: exponent of (1 - p_t); the larger, the more easy samples are down-weighted
    reduction: 'none' | 'mean' | 'sum'
    """
    inputs = inputs.float() # (V,)
    targets = targets.float() # (V,)
    p = torch.sigmoid(inputs) # (V,)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # (V,)
    p_t = p * targets + (1 - p) * (1 - targets) # (V,)
    loss = ce_loss * ((1 - p_t) ** gamma) # (V,)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets) # (V,)
        loss = alpha_t * loss # (V,)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    class_aware: bool = False,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """
    Distance-IoU loss (Zheng et al., https://arxiv.org/abs/1911.08287)

    A 1D event is a point c plus its distances to the two boundaries:
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0. Prediction and target share the point c.

    Code adapted from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    input_offsets / target_offsets: (N, 2) distances to the two boundaries
    reduction: 'none' | 'mean' | 'sum'
    eps: small number to avoid division by 0
    """
    input_offsets = input_offsets.float() # (N, 2)
    target_offsets = target_offsets.float() # (N, 2)
    # distances must be non-negative
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    if class_aware:
        #                           (B, P)                              (B, P)
        mask = torch.logical_or((target_offsets[:, :, 0] > 0), (target_offsets[:, :, 1] > 0)) # (B, P)
        input_offsets = input_offsets[mask]
        target_offsets = target_offsets[mask]

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest segment enclosing both
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between the two centres
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss, iouk


def omni_distill_loss(
    students,         # (N, dim) student vector of every positive point, L2-normalised
    teachers,         # (N, dim) teacher vector of the GT segment of every point, L2-normalised
    anchor_steps,     # (N,) step of the GT segment of every point
    anchor_keys,      # (N,) segment id of the GT segment of every point
    batch_vectors,    # (M, dim) teacher vectors of every GT segment of the batch (the in-batch negatives)
    batch_steps,      # (M,) their steps
    batch_keys,       # (M,) their segment ids
    temperature,      # softmax temperature (0.05)
):
    """InfoNCE: each student must pick its own teacher among
        - its own teacher (the positive, column 0),
        - the teachers of the other segments of the batch (the negatives).
    The anchor's own segment, and every segment with the same step as the anchor, are masked among the
    negatives: the first is the positive itself, the second is the same action in another video, not a wrong answer.
    (A mined bank of extra negatives, 4096 or 60000 of them, did not help: removed.)
    """
    # similarity to the positive: (N, 1)
    positive = (students * teachers).sum(dim=-1, keepdim=True)

    negative = students @ batch_vectors.t()                                                     # (N, M)
    same_segment = anchor_keys[:, None] == batch_keys[None, :]                                   # (N, M)
    same_step = anchor_steps[:, None] == batch_steps[None, :]                                    # (N, M)
    negative = negative.masked_fill(same_segment | same_step, float('-inf'))

    logits = torch.cat([positive, negative], dim=1) / temperature                                # (N, 1 + M)
    target = torch.zeros(len(students), dtype=torch.long, device=students.device)                # column 0
    return F.cross_entropy(logits, target)


def omni_loss(
    video_list,  # B dicts, one per video (omni_seg, omni_ok, seg_key)
    pos,         # (B, P) bool, points inside a GT segment
    gt_cls,      # (B, P, 48) one-hot slot of the GT segment of every point
    cap_pool,    # (B, 48) step (pool row) of every slot, -1 = empty slot
    omni_em,     # (B, P, 3584) student vector of every point, L2-normalised
    temperature, # softmax temperature (0.05)
):
    """Distillation of the OmniRetriever segment vectors (teacher) into the omni head (student).

    Returns two losses:
        nce : InfoNCE, the student must pick its own teacher among the batch segments
              (learns what tells segments apart, but ignores the part every teacher shares)
        cos : 1 - cosine(student, teacher), pulls the student onto its teacher, shared part included
    """
    dev = omni_em.device
    # what the dataset gives for every video of the batch, stacked: one row per slot (GT segment)
    teacher_all = torch.stack([x['omni_seg'] for x in video_list]).to(dev)   # (B, 48, 3584) teacher of every slot
    has_teacher = torch.stack([x['omni_ok'] for x in video_list]).to(dev)    # (B, 48) True = the slot has one
    seg_keys = torch.stack([x['seg_key'] for x in video_list]).to(dev)       # (B, 48) unique id of every slot

    # 1. the anchors: every point that lies inside a GT segment AND whose segment has a teacher vector
    slot = gt_cls.argmax(-1)                          # (B, P) GT segment (slot) of every point (0 outside any)
    video = torch.arange(slot.shape[0], device=dev)[:, None].expand_as(slot)   # (B, P) video of every point
    slot_has_teacher = has_teacher[video, slot]       # (B, P) [b, p] = has_teacher[b, slot[b, p]]
    is_anchor = pos & slot_has_teacher                # (B, P)
    if not is_anchor.any():
        zero = omni_em.sum() * 0.0   # no teacher in this batch: zero losses that keep the graph
        return {'nce': zero, 'cos': zero}

    # every anchor is described by three numbers: its video, its point, and the slot of its segment
    anchor_video, anchor_point = is_anchor.nonzero(as_tuple=True)   # (N,), (N,)
    anchor_slot = slot[anchor_video, anchor_point]                  # (N,)

    students = omni_em[anchor_video, anchor_point]                  # (N, 3584) what the model predicts
    teachers = teacher_all[anchor_video, anchor_slot]               # (N, 3584) what it should predict
    anchor_steps = cap_pool[anchor_video, anchor_slot]              # (N,) step of the anchor's segment
    anchor_keys = seg_keys[anchor_video, anchor_slot]               # (N,) id of the anchor's segment

    # 2. the in-batch negatives: every GT segment of the batch that has a teacher vector
    #    (the anchor's own segment and the segments of the same step are masked in omni_distill_loss)
    batch_vectors = teacher_all[has_teacher]                        # (M, 3584)
    batch_steps = cap_pool[has_teacher]                             # (M,)
    batch_keys = seg_keys[has_teacher]                              # (M,)

    # 3. the contrastive loss, against the segments of the batch
    nce_loss = omni_distill_loss(students, teachers, anchor_steps, anchor_keys,
                                 batch_vectors, batch_steps, batch_keys,
                                 temperature)

    # 4. the direct pull: cosine distance between every student and its own teacher
    cosine = (students * teachers).sum(dim=-1)   # (N,)
    cos_loss = (1.0 - cosine).mean()
    return {'nce': nce_loss, 'cos': cos_loss}


def omni_span_loss(
    students,     # (M, dim) omni span head vector of every GT segment, L2-normalised
    teachers,     # (M, dim) teacher vector of the same segments, L2-normalised
    steps,        # (M,) step of every segment
    keys,         # (M,) segment id of every segment
    temperature,  # softmax temperature (0.05)
):
    """Losses of the omni span head, one anchor per GT segment:
        nce : InfoNCE, its own teacher against the other segments of the batch (same steps masked)
        cos : 1 - cosine(student, teacher)
    """
    nce_loss = omni_distill_loss(students, teachers, steps, keys,
                                 teachers, steps, keys,      # in-batch negatives: the teachers of the batch
                                 temperature)
    cos_loss = (1.0 - (students * teachers).sum(dim=-1)).mean()
    return {'nce': nce_loss, 'cos': cos_loss}
