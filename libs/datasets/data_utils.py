import numpy as np
import torch
from torch.nn import functional as F


def trivial_batch_collator(batch):
    """
        No collation: a batch is just a list of dicts
    """
    return batch

def label_points(
    points,       # one entry per level, each (T_l, 4): (t, regression range low, high, stride)
    gt_segments,  # (n, 2) GT segments, in steps
    gt_labels,    # (n,) index of every segment within the video
    num_classes,  # at most this many GT segments per video (48)
):
        """Training targets of every pyramid point (as in ActionFormer).

        A point is positive for a GT segment if it lies inside the segment and its largest distance to the
        two boundaries falls in the regression range of its level. If several segments qualify, the shortest wins.
        Returns, with P the number of points over all levels:
          cls_targets (P, num_classes)  one-hot caption index of the segment holding the point, all 0 for background
          reg_targets (P, 2)            distances to the segment start and end, divided by the stride
        """
        # concat the points of all levels: List[T x 4] -> (F T, 4)
        concat_points = torch.cat(points, dim=0)
        num_pts = concat_points.shape[0] # F T
        num_gts = gt_segments.shape[0] # n

        # length of every segment -> F T x N
        lens = gt_segments[:, 1] - gt_segments[:, 0] # (n,)
        #       (1, n)          (F T)
        lens = lens[None, :].repeat(num_pts, 1) # (F T, n)

        # distance of every point to the two boundaries of every segment -> (F T, n, 2)
        #           (1, n, 2)               F T         n
        gt_segs = gt_segments[None].expand(num_pts, num_gts, 2)
        #           (F T, 1)                (F T, n)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0] # (F T, n); points - left
        right = gt_segs[:, :, 1] - concat_points[:, 0, None] # right - points
        reg_targets = torch.stack((left, right), dim=-1) # (F T, n, 2)

        # point inside a GT segment
        inside_gt_seg_mask = reg_targets.min(-1)[0] > 0 # (F T, n)

        # limit the regression range by the level of the point
        max_regress_distance = reg_targets.max(-1)[0] # (F T, n)
        # F T x N
        inside_regress_range = torch.logical_and(
        #           (F T, n)                (F T, 1)
            (max_regress_distance >= concat_points[:, 1, None]), # (F T, n)
            (max_regress_distance <= concat_points[:, 2, None])
        )

        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))

        # a point still in several segments takes the shortest one (easiest to regress)
        # F T x n -> (F T,)
        min_len, min_len_inds = lens.min(dim=1)
        # several segments of almost the same length
        min_len_mask = torch.logical_and(
        #   (F T, n)       (F T, 1)
            (lens <= (min_len[:, None] + 1e-3)), # (F T, n)
            (lens < float('inf'))
        ).to(reg_targets.dtype)
        

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(gt_labels, num_classes).to(reg_targets.dtype) # (n, 16)
        cls_targets = min_len_mask @ gt_label_one_hot # (F T, 16)
        # several GT segments with the same label and the same boundaries
        cls_targets.clamp_(min=0.0, max=1.0)

        # distances to the chosen segment
        reg_targets = reg_targets[range(num_pts), min_len_inds] # (F T, 2)
        # divided by the stride of the level
        #                   (F T, 1)
        reg_targets /= concat_points[:, 3, None]
        return cls_targets, reg_targets
