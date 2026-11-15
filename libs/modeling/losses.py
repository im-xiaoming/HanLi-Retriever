import torch
from torch.nn import functional as F
from torch import Tensor

from .heads import pad_segments


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


class TeacherQueue:
    """Teacher vectors of the last batches, extra negatives of the omni InfoNCE.

    The teachers are fixed vectors read from disk (not computed by the model), so an old entry is as valid as a
    new one: no momentum encoder is needed, unlike MoCo. A plain attribute of the model (not a buffer): it is
    not saved in the checkpoints.
    """

    def __init__(self, size):
        self.size = size
        self.bank = None   # (size, 3584), allocated at the first push
        self.ptr = 0       # next row to write
        self.n = 0         # filled rows

    def get(self):
        return None if self.n == 0 else self.bank[:self.n]

    @torch.no_grad()
    def push(self, x):     # (M, 3584)
        if self.size <= 0 or len(x) == 0:
            return
        if self.bank is None:
            self.bank = x.new_zeros(self.size, x.shape[1])
        x = x[-self.size:].detach()
        idx = (self.ptr + torch.arange(len(x), device=x.device)) % self.size
        self.bank[idx] = x.to(self.bank.dtype)
        self.ptr = int((self.ptr + len(x)) % self.size)
        self.n = min(self.n + len(x), self.size)


def omni_distill_loss(
    students,      # (N, 3584) student vectors, L2-normalised
    teachers,      # (M, 3584) the teacher vectors of the batch, L2-normalised
    target,        # (N,) for every student, the row of `teachers` that is its own teacher
    temperature,   # softmax temperature of the kd (0.05)
    bank=None,     # (Q, 3584) teachers of earlier batches (TeacherQueue), extra negatives of the nce, or None
    nce_temperature=0.05,
):
    """Distillation of the OmniRetriever vectors (teachers) into the model (students). Three losses:

    kd  : KL distillation. Every vector (student or teacher) gets a similarity distribution over the teachers of
          the batch: softmax(similarities / temperature). The student's distribution must match the one of its own
          teacher. So the student learns how its segment relates to the other segments, as the teacher sees it
          (a segment of the same step, similar to its teacher, gets a high probability too).
    cos : 1 - cosine(student, own teacher). Pulls every student onto its teacher, including the part that every
          teacher shares (the kd ignores it), so the student vector really is close to the teacher vector.
    nce : InfoNCE, student -> teachers: its own teacher must be the closest among the teachers of the batch and of
          the queue (one-hot target, where kd copies the teacher's soft distribution). This is what OmniR@1
          measures (own teacher = closest of all validation teachers). A candidate (almost) equal to the own
          teacher (cos > 0.999: the same segment again) is not a negative.
    """
    if len(students) == 0:   # no segment with a teacher in this batch (very rare): no loss
        zero = students.sum() * 0.0
        return {'kd': zero, 'cos': zero, 'nce': zero}

    own_teachers = teachers[target]                                                  # (N, 3584)

    student_logits = students @ teachers.t() / temperature                           # (N, M)
    teacher_logits = own_teachers @ teachers.t() / temperature                       # (N, M)
    kd_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.log_softmax(teacher_logits, dim=-1),
                       log_target=True, reduction='batchmean')

    cos_loss = (1.0 - F.cosine_similarity(students, own_teachers, dim=-1)).mean()

    cands = teachers if bank is None else torch.cat((teachers, bank.to(teachers.dtype)))   # (M + Q, 3584)
    logits = students @ cands.t() / nce_temperature                                  # (N, M + Q)
    with torch.no_grad():
        same = own_teachers @ cands.t() > 0.999                                      # (N, M + Q) duplicates
        same[torch.arange(len(target), device=target.device), target] = False       # but not the target itself
    nce_loss = F.cross_entropy(logits.masked_fill(same, float('-inf')), target)
    return {'kd': kd_loss, 'cos': cos_loss, 'nce': nce_loss}


def omni_losses(
    span,
    video_list,  # B dicts: omni_seg (48, 3584) teacher of every GT segment, omni_ok (48,) True = it has one,
                    #          segments / labels = the GT segments left after the crop (labels = their slot)
    pos,         # (B, P) bool, points inside a GT segment
    gt_cls,      # (B, P, 48) one-hot: the GT segment (slot) of every point
    omni_em,     # (B, P, 3584) point head: student vector of every point, L2-normalised
    feats,       # per level (B, D, T_l) pyramid features, read by the span head
    lens,        # (B, L) real steps of every level
    temperature=0.05,
    queue=None,      # TeacherQueue: teachers of earlier batches, extra negatives of the nce; the batch is added after
    nce_temperature=0.05,
):
    """Distillation of the OmniRetriever vectors (teachers), for the point head and the span head.

    1. The teachers of the batch: every GT segment that has a teacher vector, put in one list (M rows).
        teacher_row[b, s] remembers where segment s of video b sits in that list (-1: no teacher).
    2. Point head: every point inside a GT segment that has a teacher is a student; its target is the row of
        its segment's teacher.
    3. Span head: every GT segment (left after the crop) that has a teacher gets one student vector from the
        span head; its target is the row of its own teacher.
    Both are trained by omni_distill_loss (KL distillation over the M teachers, cosine to the own teacher and
    InfoNCE over the M teachers + the queue).
    """
    dev = omni_em.device

    # 1. the teachers of the batch
    omni_seg = torch.stack([x['omni_seg'] for x in video_list]).to(dev)    # (B, 48, 3584)
    omni_ok = torch.stack([x['omni_ok'] for x in video_list]).to(dev)      # (B, 48)
    teachers = omni_seg[omni_ok]                                            # (M, 3584)
    teacher_row = torch.full(omni_ok.shape, -1, dtype=torch.long, device=dev)   # (B, 48)
    teacher_row[omni_ok] = torch.arange(len(teachers), device=dev)

    # 2. point head
    video, point = pos.nonzero(as_tuple=True)          # (N',) video and point of every point inside a segment
    segment = gt_cls[video, point].argmax(-1)           # (N',) the GT segment (slot) the point lies in
    target = teacher_row[video, segment]                # (N',) row of its teacher, -1 = no teacher
    keep = target >= 0
    students = omni_em[video[keep], point[keep]]        # (N, 3584)
    bank = None if queue is None else queue.get()
    point_losses = omni_distill_loss(students, teachers, target[keep], temperature, bank, nce_temperature)

    # 3. span head
    span_segs, span_target = [], []
    for b, x in enumerate(video_list):
        target = teacher_row[b, x['labels'].to(dev)]    # (n,) row of the teacher of every GT segment left
        keep = target >= 0
        span_segs.append(x['segments'].to(dev).float()[keep])   # (n', 2) start and end, in level-0 steps
        span_target.append(target[keep])
    segs, real = pad_segments(span_segs)                # (B, n, 2), (B, n): the whole batch in one call
    span_students = span(feats, lens, segs)[real]       # (N, 3584), in the order of span_target
    span_losses = omni_distill_loss(span_students, teachers, torch.cat(span_target), temperature,
                                    bank, nce_temperature)
    if queue is not None:
        queue.push(teachers)
    return point_losses, span_losses