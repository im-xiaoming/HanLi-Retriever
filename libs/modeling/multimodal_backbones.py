import torch
from torch import nn
from torch.nn import functional as F

from .models import register_multimodal_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock,
                    MaskedConv1D, LayerNorm, TemporalMaxer, DynELayer)


@register_multimodal_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    def __init__(
        self,
        n_in_V,                # input video channels
        n_in_A,                # input audio channels
        n_embd,                # channels after the conv embedding
        n_head,                # attention heads in the transformer
        n_embd_ks,             # kernel size of the conv embedding
        max_len,               # maximum sequence length
        arch = (2, 3, 5),      # (#conv, #transformer in the stem, #transformer in the pyramid branch)
        scale_factor = 2,      # length ratio between two levels
        with_ln = False,       # LayerNorm after the convs
        attn_pdrop = 0.0,      # dropout on the attention matrix
        proj_pdrop = 0.0,      # dropout of the projection / MLP
        path_pdrop = 0.0,      # drop path rate
        use_abs_pe = False,    # absolute position embedding
    ):
        super().__init__()
        self.arch = arch
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe

        # position embedding (1, C, T), divided by sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # conv embedding network
        # 768 -> 512
        self.embd_V = nn.ModuleList()
        self.embd_A = nn.ModuleList()
        self.embd_norm_V = nn.ModuleList()
        self.embd_norm_A = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels_V = n_in_V
                in_channels_A = n_in_A
            else:
                in_channels_V = n_embd
                in_channels_A = n_embd
            self.embd_V.append(MaskedConv1D(
                    in_channels_V, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            self.embd_A.append(MaskedConv1D(
                    in_channels_A, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm_V.append(
                    LayerNorm(n_embd)
                )
                self.embd_norm_A.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm_V.append(nn.Identity())
                self.embd_norm_A.append(nn.Identity())

        # stem: plain transformer (self-attention)
        self.self_att_V = nn.ModuleList()
        self.self_att_A = nn.ModuleList()

        for idx in range(arch[1]-1): 
            self.self_att_V.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.self_att_A.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
        # cross-attention at the original time resolution
        self.ori_cross_att_Va = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
        self.ori_cross_att_Av = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )

        # cross-attention after every downsampling
        self.cross_att_Va = nn.ModuleList()
        self.cross_att_Av = nn.ModuleList()
        for idx in range(arch[2]):
            self.cross_att_Va.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.cross_att_Av.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )

        # weight init
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # zero the bias of nn.Linear / nn.Conv1d
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x_V, x_A, mask):
        # x_V / x_A: (batch, channels, sequence length)
        # mask: (batch, 1, sequence length), bool
        B, _, T = x_V.size()
        mask_V = mask_A = mask
        # embedding network
        for idx in range(len(self.embd_V)):
            x_V, mask_V = self.embd_V[idx](x_V, mask_V) 
            x_V = self.relu(self.embd_norm_V[idx](x_V))

            x_A, mask_A = self.embd_A[idx](x_A, mask_A)
            x_A = self.relu(self.embd_norm_A[idx](x_A))

        # fixed-length position embedding
        if self.use_abs_pe and self.training:
            pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)

        # inference: resample the position embedding if the sequence is longer than max_len
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)

        for idx in range(len(self.self_att_V)):
            x_V, mask_V = self.self_att_V[idx](x_V, x_V, mask_V)
            x_A, mask_A = self.self_att_A[idx](x_A, x_A, mask_A)

        # level 0: x_Va = V updated by audio queries, x_Av = A updated by video queries
        x_Va, mask_V = self.ori_cross_att_Va(x_V, x_A, mask_V)
        x_Av, mask_A = self.ori_cross_att_Av(x_A, x_V, mask_A)

        # prepare the outputs
        out_feats_V = tuple()
        out_feats_A = tuple()
        out_masks_V = tuple()
        # original resolution
        out_feats_V += (x_Va, )
        out_masks_V += (mask_V, )
        out_feats_A += (x_Av, )

        # main branch with downsampling: level idx + 1 is built from level idx of both streams
        for idx in range(len(self.cross_att_Va)):
            x_V, mask_V = self.cross_att_Va[idx](out_feats_V[idx], out_feats_A[idx], mask_V)
            x_A, mask_A = self.cross_att_Av[idx](out_feats_A[idx], out_feats_V[idx], mask_A)

            out_feats_V += (x_V, )
            out_masks_V += (mask_V, )
            out_feats_A += (x_A, )

        return out_feats_V, out_feats_A, out_masks_V

@register_multimodal_backbone("convPooler")
class ConvPoolerBackbone(nn.Module):
    """
        TemporalMaxer backbone (Tang et al. 2023): conv embedding, then arch[2] max-pooling steps that each
        halve the length. No transformer and no attention; video and audio stay two separate streams
        (the heads see both, concatenated). arch[1] and the attention arguments are not used.
    """
    def __init__(
        self,
        n_in_V,                # input video channels
        n_in_A,                # input audio channels
        n_embd,                # channels after the conv embedding
        n_embd_ks,             # kernel size of the conv embedding
        arch = (2, 3, 5),      # (#conv, unused, #pooling levels)
        scale_factor = 2,      # length ratio between two levels
        with_ln = False,       # LayerNorm after the convs
        **kwargs,              # the convTransformer arguments this backbone does not use
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.embd_V, self.embd_A = nn.ModuleList(), nn.ModuleList()
        self.embd_norm_V, self.embd_norm_A = nn.ModuleList(), nn.ModuleList()
        for idx in range(arch[0]):
            c_V, c_A = (n_in_V, n_in_A) if idx == 0 else (n_embd, n_embd)
            self.embd_V.append(MaskedConv1D(c_V, n_embd, n_embd_ks, stride=1, padding=n_embd_ks//2, bias=(not with_ln)))
            self.embd_A.append(MaskedConv1D(c_A, n_embd, n_embd_ks, stride=1, padding=n_embd_ks//2, bias=(not with_ln)))
            self.embd_norm_V.append(LayerNorm(n_embd) if with_ln else nn.Identity())
            self.embd_norm_A.append(LayerNorm(n_embd) if with_ln else nn.Identity())
        self.branch = nn.ModuleList([TemporalMaxer(3, scale_factor, 1) for _ in range(arch[2])])
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # zero the bias of nn.Linear / nn.Conv1d
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x_V, x_A, mask):
        # x_V / x_A: (batch, channels, sequence length); mask: (batch, 1, sequence length), bool
        mask_V = mask_A = mask
        for idx in range(len(self.embd_V)):
            x_V, mask_V = self.embd_V[idx](x_V, mask_V)
            x_V = self.relu(self.embd_norm_V[idx](x_V))
            x_A, mask_A = self.embd_A[idx](x_A, mask_A)
            x_A = self.relu(self.embd_norm_A[idx](x_A))

        out_feats_V, out_feats_A, out_masks_V = (x_V, ), (x_A, ), (mask_V, )
        for pool in self.branch:
            x_V, mask_V = pool(x_V, mask_V)
            x_A, _ = pool(x_A, mask_A)
            mask_A = mask_V
            out_feats_V += (x_V, )
            out_feats_A += (x_A, )
            out_masks_V += (mask_V, )
        return out_feats_V, out_feats_A, out_masks_V


@register_multimodal_backbone("dynE")
class DynEBackbone(ConvTransformerBackbone):
    """
        DyFADet encoder (Yang et al. 2024) in the two-stream layout: the self-attention blocks of the stem and the
        downsampling cross-attention blocks of the pyramid become DynE layers; the cross-attention between video and
        audio at the original resolution stays, so the two streams still exchange information once.
    """
    def __init__(self, n_in_V, n_in_A, n_embd, n_head, n_embd_ks, max_len, arch=(2, 3, 5), scale_factor=2,
                 with_ln=False, attn_pdrop=0.0, proj_pdrop=0.0, path_pdrop=0.0, use_abs_pe=False,
                 dyn_k=5, dyn_mlp_dim=768):
        super().__init__(n_in_V, n_in_A, n_embd, n_head, n_embd_ks, max_len, arch, scale_factor, with_ln,
                         attn_pdrop, proj_pdrop, path_pdrop, use_abs_pe)
        self.dyn_stem_V = nn.ModuleList([DynELayer(n_embd, 1, 1, dyn_k, dyn_mlp_dim) for _ in range(arch[1] - 1)])
        self.dyn_stem_A = nn.ModuleList([DynELayer(n_embd, 1, 1, dyn_k, dyn_mlp_dim) for _ in range(arch[1] - 1)])
        self.dyn_branch_V = nn.ModuleList([DynELayer(n_embd, 1, scale_factor, dyn_k, dyn_mlp_dim, path_pdrop)
                                           for _ in range(arch[2])])
        self.dyn_branch_A = nn.ModuleList([DynELayer(n_embd, 1, scale_factor, dyn_k, dyn_mlp_dim, path_pdrop)
                                           for _ in range(arch[2])])
        # the attention blocks replaced above
        del self.self_att_V, self.self_att_A, self.cross_att_Va, self.cross_att_Av

    def forward(self, x_V, x_A, mask):
        T = x_V.size(-1)
        mask_V = mask_A = mask
        for idx in range(len(self.embd_V)):
            x_V, mask_V = self.embd_V[idx](x_V, mask_V)
            x_V = self.relu(self.embd_norm_V[idx](x_V))
            x_A, mask_A = self.embd_A[idx](x_A, mask_A)
            x_A = self.relu(self.embd_norm_A[idx](x_A))
        if self.use_abs_pe:
            pe = self.pos_embd if (self.training or T < self.max_len) else \
                F.interpolate(self.pos_embd, T, mode='linear', align_corners=False)
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)
        for blk_V, blk_A in zip(self.dyn_stem_V, self.dyn_stem_A):
            x_V, mask_V = blk_V(x_V, mask_V)
            x_A, mask_A = blk_A(x_A, mask_A)
        # level 0: the one exchange between the two streams
        x_V, mask_V = self.ori_cross_att_Va(x_V, x_A, mask_V)
        x_A, mask_A = self.ori_cross_att_Av(x_A, x_V, mask_A)
        out_feats_V, out_feats_A, out_masks_V = (x_V, ), (x_A, ), (mask_V, )
        for blk_V, blk_A in zip(self.dyn_branch_V, self.dyn_branch_A):
            x_V, mask_V = blk_V(x_V, mask_V)
            x_A, mask_A = blk_A(x_A, mask_A)
            out_feats_V += (x_V, )
            out_feats_A += (x_A, )
            out_masks_V += (mask_V, )
        return out_feats_V, out_feats_A, out_masks_V
