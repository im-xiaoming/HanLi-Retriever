import copy
import random
import numpy as np
import torch
from torch.nn import functional as F


def trivial_batch_collator(batch):
    """
        No collation: a batch is just a list of dicts
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reseed the random generators of every DataLoader worker
    """
    seed = torch.initial_seed() % (2 ** 31)
    np.random.seed(seed)
    random.seed(seed)

def truncate_feats(
    data_dict,           # dict of one video (see below)
    max_seq_len,         # maximum length (steps)
    trunc_thresh,        # a GT segment is kept if at least this fraction of it lies in the window
    offset,              # 0.5: step i covers [i - 0.5, i + 0.5]
    crop_ratio,          # (0.9, 1.0): the window covers 90% to 100% of the video
    max_num_trials=200,  # maximum attempts to find a window that keeps at least one segment
):
    """
    Training augmentation: crop a random window of the features and cut the GT segments to it.

    The window is crop_ratio[0] to crop_ratio[1] of the video long. A segment is kept if at least trunc_thresh
    of it lies in the window. Up to max_num_trials attempts until at least one segment is left.

    data_dict = {'video_id'        : str
                 'feats'           : {'visual': Tensor C x T, 'audio': Tensor C x T}
                 'segments'        : Tensor N x 2 (in steps)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : float
                 'feat_num_frames' : float}
    """
    # get the meta info; feats = {'visual': C x T, 'audio': C x T}
    feat_len = data_dict['feats']['visual'].shape[1] # T=256

    num_segs = data_dict['segments'].shape[0] # n

    # random window length in [l, r]
    assert crop_ratio[0] <= crop_ratio[1], "crop_ratio[0] must be <= crop_ratio[1]"
    max_seq_len = random.randint(
        max(round(crop_ratio[0] * feat_len), 1),
        min(round(crop_ratio[1] * feat_len), feat_len),
    )
    # a window as long as the video needs no crop
    if feat_len == max_seq_len:
        return data_dict

    # otherwise copy the dict before editing it
    data_dict = copy.deepcopy(data_dict)

    # try a few times until the window keeps at least one segment
    for _ in range(max_num_trials):

        # random window position
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.tensor([st, ed], dtype=torch.float32) # (2,)

        # intersection of the window with every segment
        #              (1, 2)       (n, 1)
        window = window[None].repeat(num_segs, 1) # (n, 2)
        left = torch.maximum(window[:, 0] - offset, data_dict['segments'][:, 0])
        right = torch.minimum(window[:, 1] + offset, data_dict['segments'][:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs

        # keep the segments whose intersection passes the threshold
        seg_idx = (inter_ratio >= trunc_thresh)

        # stop once at least one segment is left
        if seg_idx.sum().item() > 0:
            break

    data_dict['feats']['visual'] = data_dict['feats']['visual'][:, st:ed].clone()
    data_dict['feats']['audio'] = data_dict['feats']['audio'][:, st:ed].clone()
    # segments: N x 2, in steps
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1) # (n, 2)
    # shift the time stamps to the window start
    data_dict['segments'] = data_dict['segments'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict


def label_points(
    points,       # one entry per level, each (T_l, 4): (t, regression range low, high, stride)
    gt_segments,  # (n, 2) GT segments, in steps
    gt_labels,    # (n,) index of every segment within the video
    num_classes,  # at most this many GT segments per video (48)
    center_sample_radius: float = 0.0,  # > 0: positive only within this many strides of the centre (ActionFormer)
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
        if center_sample_radius > 0:
            # centre sampling: only the points near the segment centre, radius in strides of the point's level
            ctr = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1]) # (F T, n)
            r = center_sample_radius * concat_points[:, 3, None] # (F T, 1)
            inside_gt_seg_mask &= (concat_points[:, 0, None] - ctr).abs() <= r

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
