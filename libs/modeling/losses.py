import torch
from torch.nn import functional as F
from torch import Tensor

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


def quality_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Quality focal loss of GFL (Li et al., https://arxiv.org/abs/2006.04388): focal loss with soft targets.

    inputs: logits, any shape
    targets: soft labels in [0, 1], same shape as inputs (IoU of the predicted segment for positives, 0 elsewhere)
    beta: exponent of |target - p|; the closer the prediction, the smaller the weight
    reduction: 'none' | 'mean' | 'sum'
    """
    inputs = inputs.float() # (V,)
    targets = targets.float() # (V,)
    p = torch.sigmoid(inputs) # (V,)
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") * (targets - p).abs() ** beta

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
