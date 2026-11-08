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

    return loss


def omni_distill_loss(
    students,      # (N, 3584) student vectors, L2-normalised
    teachers,      # (M, 3584) the teacher vectors of the batch, L2-normalised
    target,        # (N,) for every student, the row of `teachers` that is its own teacher
    temperature,   # softmax temperature (0.05)
):
    """Distillation of the OmniRetriever vectors (teachers) into the model (students). Two losses:

    kd  : KL distillation. Every vector (student or teacher) gets a similarity distribution over the teachers of
          the batch: softmax(similarities / temperature). The student's distribution must match the one of its own
          teacher. So the student learns how its segment relates to the other segments, as the teacher sees it
          (a segment of the same step, similar to its teacher, gets a high probability too).
    cos : 1 - cosine(student, own teacher). Pulls every student onto its teacher, including the part that every
          teacher shares (the kd ignores it), so the student vector really is close to the teacher vector.
    """
    if len(students) == 0:   # no segment with a teacher in this batch (very rare): no loss
        zero = students.sum() * 0.0
        return {'kd': zero, 'cos': zero}

    own_teachers = teachers[target]                                                  # (N, 3584)

    student_logits = students @ teachers.t() / temperature                           # (N, M)
    teacher_logits = own_teachers @ teachers.t() / temperature                       # (N, M)
    kd_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.log_softmax(teacher_logits, dim=-1),
                       log_target=True, reduction='batchmean')

    cos_loss = (1.0 - F.cosine_similarity(students, own_teachers, dim=-1)).mean()
    return {'kd': kd_loss, 'cos': cos_loss}
