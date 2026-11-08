import numpy as np
from pathlib import Path
import os
import torch.nn.functional as F
import torch

def load_step_vectors(
    path,      # npz with 'ids' (global_step_index) and 'emb' (N, D), tools/htstep_data.py build
    step_row,  # {global_step_index: pool row}
):
    """(4958, D) vectors in pool order."""
    z = np.load(path)
    emb = np.zeros((len(step_row), z['emb'].shape[1]), dtype=np.float32)
    emb[[step_row[int(i)] for i in z['ids']]] = z['emb']
    return emb


def load_omni_segments(
    path,           # <omni folder>/<video_id>.npz, or a path that does not exist (video without omni vectors)
    ann_slots,      # {index in the json 'annotations' list: slot of that segment in this video (0 .. n-1)}
    num_classes=48,
    dim=3584,       # size of an OmniRetriever vector
):
    """Teacher vector of every GT segment of one video, in slot order: (48, 3584) vectors, (48,) ok.
    The file has one row per encoded segment: emb (k, 3584), seg_idx (k,) index in the json annotations."""
    vectors = np.zeros((num_classes, dim), dtype=np.float32)
    ok = np.zeros((num_classes,), dtype=bool)
    if not os.path.exists(path):
        return vectors, ok

    z = np.load(path)
    for emb, seg_idx in zip(z['emb'], z['seg_idx']):
        if int(seg_idx) not in ann_slots:
            continue   # a segment the dataset dropped (it starts after the features end)
        slot = ann_slots[int(seg_idx)]
        vector = emb.astype(np.float32)
        vectors[slot] = vector / max(float(np.linalg.norm(vector)), 1e-6)
        ok[slot] = True
    return vectors, ok


def load_normalize(feat_folder, vid_id):
    z = np.load(os.path.join(feat_folder, vid_id))
    fv = F.normalize(torch.from_numpy(z['k768'].astype(np.float32)), p=2, dim=-1, eps=1e-6)  # (T, 768)
    fa = F.normalize(torch.from_numpy(z['a768'].astype(np.float32)), p=2, dim=-1, eps=1e-6)
    return fv, fa


@torch.no_grad()
def preprocessing(
    video_list,        # B dicts, one per video (HTStepDataset.__getitem__)
    padding_val,   # value of the padding steps,
    max_seq_len,
    device
):
    """Stack the videos of the batch into (B, C=768, T) tensors and a (B, 1, T) mask.
    T is always max_seq_len: videos cropped during training are padded."""

    fv = [x['feats']['visual'] for x in video_list] # x['feats']['visual'] = (C, T')
    fa = [x['feats']['audio'] for x in video_list]
    lens = torch.as_tensor([f.shape[-1] for f in fv]) # T are different because of data agumnetation
    max_len = max_seq_len

    def pad(xs):
        """Padding function"""
        # new_full: create tensor same device and dtype
        #                        B           C           T
        out = xs[0].new_full([len(xs), xs[0].shape[0], max_len], padding_val)
        for s, d in zip(xs, out):
            d[..., :s.shape[-1]].copy_(s)
        return out.to(device)

    # B: bactch size
    masks = (torch.arange(max_len)[None, :] < lens[:, None]).unsqueeze(1).to(device) # (1, T) < (B, 1) = (B, T) -> (B, 1, T)
    return pad(fv), pad(fa), masks # (B, C, T)