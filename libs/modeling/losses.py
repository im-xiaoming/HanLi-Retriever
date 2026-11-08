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

    # ---- 0. what the dataset gives, stacked over the B videos of the batch -------------------------------------
    # Every video has 48 slots, slot s = its s-th GT segment (the same number as its label in gt_cls).
    #   teacher_all[b, s] : OmniRetriever vector (the teacher) of segment s of video b, zeros if it has none
    #   has_teacher[b, s] : True if that segment has a teacher vector (some segments could not be encoded,
    #                       and the slots after the last segment of the video are empty)
    #   seg_keys[b, s]    : a number unique to that segment in the whole split, used below to recognise
    #                       "the anchor's own segment" among the negatives
    teacher_all = torch.stack([x['omni_seg'] for x in video_list]).to(dev)   # (B, 48, 3584)
    has_teacher = torch.stack([x['omni_ok'] for x in video_list]).to(dev)    # (B, 48)
    seg_keys = torch.stack([x['seg_key'] for x in video_list]).to(dev)       # (B, 48)

    # ---- 1. the anchors: every positive point whose GT segment has a teacher vector ---------------------------
    # pos (B, P) marks the points that lie inside a GT segment. Indexing a tensor with pos (omni_em[pos],
    # gt_cls[pos], ...) keeps those points only and lays them out as one list of N' rows, always in the same
    # order: video 0 first (points in time order), then video 1, and so on. Every list built below follows that
    # order, so row k of each of them talks about the same point.
    #
    # video_of_point: pos.nonzero gives the coordinates (video, point) of every True cell of pos;
    #                 [0] keeps the first coordinate, the video the point belongs to.
    video_of_point = pos.nonzero(as_tuple=True)[0]          # (N',) video of every positive point
    # slot_of_point: gt_cls[pos] is the one-hot row (48 columns) of every positive point;
    #                argmax gives the column holding the 1, i.e. the GT segment the point lies in.
    slot_of_point = gt_cls[pos].argmax(-1)                  # (N',) GT segment (slot) of every positive point
    # point_has_teacher: for every positive point, look up has_teacher at (its video, its segment).
    #                    True = the point can be trained (its segment has a teacher), False = skip it.
    point_has_teacher = has_teacher[video_of_point, slot_of_point]   # (N',)
    if not point_has_teacher.any():
        zero = omni_em.sum() * 0.0   # no teacher in this batch: zero losses that keep the graph
        return {'nce': zero, 'cos': zero}

    # Keep only the points that have a teacher. The same mask filters every list, so they stay aligned:
    # students[i], anchor_video[i], anchor_slot[i] all describe the same point i.
    students = omni_em[pos][point_has_teacher]                                     # (N, 3584) model output
    anchor_video = video_of_point[point_has_teacher]                               # (N,) its video
    anchor_slot = slot_of_point[point_has_teacher]                                 # (N,) its segment
    
    # With (video, segment) of every anchor, read what belongs to its segment:
    #   teachers     : the vector the student must reproduce (the right answer)
    #   anchor_steps : the step of the segment, to mask the negatives of the same step (same action elsewhere)
    #   anchor_keys  : the id of the segment, to mask the anchor's own segment among the negatives
    teachers = teacher_all[anchor_video, anchor_slot]                              # (N, 3584)
    anchor_steps = cap_pool[anchor_video, anchor_slot]                             # (N,)
    anchor_keys = seg_keys[anchor_video, anchor_slot]                              # (N,)

    # ---- 2. the in-batch negatives: every GT segment of the batch that has a teacher vector ------------------
    # teacher_all[has_teacher] keeps the (video, slot) cells that have a teacher: M segments in total.
    # These are the choices every student is compared with; its own segment is among them and is masked
    # in omni_distill_loss (via the keys), as are the segments of the same step (via the steps).
    batch_vectors = teacher_all[has_teacher]                                       # (M, 3584)
    batch_steps = cap_pool[has_teacher]                                            # (M,)
    batch_keys = seg_keys[has_teacher]                                             # (M,)

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
