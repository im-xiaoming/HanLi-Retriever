import torch
from torch import nn

from .datasets import register_generator



class BufferList(nn.Module):
    """
    Like nn.ParameterList, but for buffers

    Taken from https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/anchor_generator.py
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # the buffers are not saved in the checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

@register_generator('point')
class PointGenerator(nn.Module):
    def __init__(
        self,
        regression_range: list[list[int]],
        max_seq_len_ori: int = 256,    # maximum sequence length the points are precomputed for
        max_buffer_len_factor: float = 1.0,  # precompute max_seq_len_ori * this factor steps
        fpn_levels: int = 6,         # number of pyramid levels
        scale_factor: int = 2,       # length ratio between two levels
        max_div_factor: int = 32,     # largest stride, lengths must be divisible by it
        use_offset=False    # shift the points to the middle of their grid cell
    ):
        super().__init__()
        # check the number of levels and divisibility
        assert len(regression_range) == fpn_levels
        max_seq_len = max_seq_len_ori * max_buffer_len_factor
        assert max_seq_len % scale_factor**(fpn_levels - 1) == 0

        # save the parameters
        self.max_seq_len_ori = max_seq_len_ori
        self.max_seq_len = max_seq_len
        self.fpn_levels = fpn_levels
        self.scale_factor = scale_factor
        self.regression_range = regression_range
        self.use_offset = use_offset
        self.max_div_factor = max_div_factor

        # precompute all the points
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        points_list = []
        # loop over the pyramid levels
        for l in range(self.fpn_levels):
            stride = self.scale_factor ** l
            reg_range = torch.tensor(self.regression_range[l], dtype=torch.float32) # (2,)
            
            fpn_stride = torch.tensor(stride, dtype=torch.float32)
            points = torch.arange(0, self.max_seq_len, stride)[:, None] # (N, 1)
            
            # shift the points if asked (the current model does not)
            if self.use_offset:
                points += 0.5 * stride
            # append the regression range and the stride to every point
            #               (1, 2)          (N, 2)
            reg_range = reg_range[None].repeat(points.shape[0], 1) # (N, 2)
            #               (1, 1)          (N, 1)
            fpn_stride = fpn_stride[None].repeat(points.shape[0], 1) # (N, 1)
            # size: T x 4 (t, regression range, stride)
            #                                       (N, 4)
            points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1)) # (N, 4)

        return BufferList(points_list)

    def forward(self, fpn_strides) -> list[torch.Tensor]:
        """Points of every level: a list of (max_seq_len / stride, 4) tensors (every video is resampled to
        max_seq_len steps)."""
        pts_list = []
        max_len = self.max_seq_len_ori
        feat_lens = [int(max_len/stride) for stride in fpn_strides] # [256, 128, 64, 32, 16, 8]
        for feat_len, buffer_pts in zip(feat_lens, self.buffer_points):
            pts = buffer_pts[:feat_len, :]
            pts_list.append(pts)
        return pts_list

