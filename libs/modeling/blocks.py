import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


def Linear(in_features, out_features, bias=True):
    """nn.Linear with Xavier-uniform weights and zero bias."""
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class MaskedConv1D(nn.Module):
    """
    Masked Conv1D: the output at padding steps is set to 0. Called like Conv1d,
    but only a subset of 1D convs is supported.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # odd kernel and padding = kernel // 2 so the output lines up with the input
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero bias if any
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: (batch, channels, sequence length)
        # mask: (batch, 1, sequence length), bool
        B, C, T = x.size()
        # the input length must be divisible by the stride
        assert T % self.stride == 0
        # conv
        out_conv = self.conv(x)
        # mask of the output
        if self.stride > 1:
            # downsample the mask with nearest neighbour
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=T//self.stride,
                mode='nearest'
            )
        else:
            # keep the mask as it is
            out_mask = mask.to(x.dtype)

        if out_mask.shape[-1] != out_conv.shape[-1]:
            out_mask = out_mask[:, None, None]
        # mask the output, the mask needs no gradient
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm for (B, C, T) tensors: normalises over the channel axis C
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        # normalise over the channel axis C
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)
        # scale and shift
        if self.affine:
            out *= self.weight
            out += self.bias

        return out

# helper for the Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # returns a (1, C, T) tensor
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)

class MaskedMHCA(nn.Module):
    """
    Multi-head attention with convs and masks (from ActionFormer)

    Compared with plain MHA there is a depthwise conv on the query, key and value. The conv
    (1) encodes relative positions (instead of a position encoding);
    (2) downsamples the features when needed (stride 2 -> pyramid levels);
    (3) matches the channel count.

    As implemented, the downsampled features line up with every (s+1)-th step
    (s is the stride), which makes interpolating the matching position embedding easy.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # output channels
        n_head,          # attention heads
        n_qx_stride=1,   # downsampling stride of the query and the input
        n_kv_stride=1,   # downsampling stride of the key and value
        attn_pdrop=0.0,  # dropout on the attention matrix
        proj_pdrop=0.0,  # dropout of the output projection
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv / pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # depthwise conv of the query. Its stride is n_kv_stride as in the original; the two values
        # are always equal here, so it does not matter
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1D depthwise conv
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # depthwise conv of the key and value
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1D depthwise conv
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections of all heads
        # no mask here, the mask is applied to the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization (dropout)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x1, x2, mask):
        # x1: stream of the key and value; x2: stream of the query (x1 is x2 for self-attention).
        # The output has one row per query step.
        # x: (batch, channels, sequence length)
        # mask: (batch, 1, sequence length), bool
        B, C, T = x1.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x2, mask)
        q = self.query_norm(q)
        # key, value convs -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x1, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x1, mask)
        v = self.value_norm(v)

        # project to q, k, v
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move the head axis next to the batch axis
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)

        if T == mask.shape[-1]:
            # q must not look at padding steps
            att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
            # softmax over the attention
            att = F.softmax(att, dim=-1)
        else:
            att = F.softmax(att, dim=-1)
            att = att * kv_mask[:, :, :, None].to(att.dtype)

        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # concat the outputs of the heads
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection, masked
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)

        return out, qx_mask

class TransformerBlock(nn.Module):
    """
    Simple Transformer block (LayerNorm before the attention and the MLP)
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        n_embd,                # input channels
        n_head,                # attention heads
        n_ds_strides=(1, 1),   # downsampling strides of (q and x, k and v)
        n_out=None,            # output channels, None = input channels
        n_hidden=None,         # hidden channels of the MLP
        act_layer=nn.GELU,     # MLP activation, GELU by default
        attn_pdrop=0.0,        # dropout on the attention matrix
        proj_pdrop=0.0,        # dropout of the projection / MLP
        path_pdrop=0.0,        # drop path rate
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # LayerNorm for (B C T) tensors
        self.ln11 = LayerNorm(n_embd)
        self.ln12 = LayerNorm(n_embd)
        self.n_embd = n_embd

        self.attn = MaskedMHCA(
            n_embd,
            n_head,
            n_qx_stride=n_ds_strides[0],
            n_kv_stride=n_ds_strides[1],
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop
            )

        # skip branch: max pooling downsamples it when stride > 1
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two-layer MLP
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        # feed-forward network. The original UniAV has three, one per task (TAL, AVEL, SED); only the TAL one is used here
        self.ln2 = LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x1, x2, mask, pos_embd=None):
        # x1: the updated stream (key, value and residual branch); x2: the query stream
        # (x1 itself for self-attention): out = x1 + Attn(q=x2, k=x1, v=x1), then the MLP.
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(self.ln11(x1), self.ln12(x2), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x1) * out_mask_float + self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)

        # add pos_embd if given
        if pos_embd is not None:
            out += pos_embd * out_mask_float

        return out, out_mask

class Scale(nn.Module):
    """
    Multiply the regression output by a learnable constant
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value of the constant
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale

# the code below is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic depth: drop the whole residual branch of each sample with probability drop_prob.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # works with tensors of any rank, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop path (stochastic depth) per sample, on the main path of a residual block."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop path per sample with a per-channel scale (initialised very small, 1e-4)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):

        return drop_path(self.scale * x, self.drop_prob, self.training)

