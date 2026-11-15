"""Ridge regression from the input features of a segment to its OmniRetriever vector (teacher), combined with
the omni heads at inference.

A linear map of the raw k400 + BEATs features of a segment already reaches OmniCosC ~69.5 on val_seen, more than
the omni heads alone (~66-68), so the two are combined (postprocess.omni_gt_vectors). Fitted once on the train
segments (closed form, no training), before the model is evaluated."""
import numpy as np
import torch

RIDGE_LAMBDA = 3000.0   # 3000: OmniCosC 69.52 on val_seen vs 69.31 at 1000 and 67.70 at 100


def segment_inputs(
    visual,    # (768, T) k400 video features of one video
    audio,     # (768, T) BEATs audio features of the same steps
    segments,  # (n, 2) segments (start, end) in steps
):
    """Input of the ridge for every segment: mean and max of the features over its steps, and the mean over the
    whole video (the context). Returns (n, 3 * 1536)."""
    f = torch.cat([visual, audio], dim=0).float()                 # (1536, T)
    T = f.shape[1]
    glob = f.mean(1)
    rows = []
    for s, e in segments.tolist():
        a = int(np.clip(np.floor(s), 0, T - 1))
        b = int(np.clip(np.ceil(e) + 1, a + 1, T))                 # one past the last step, at least one step
        seg = f[:, a:b]
        rows.append(torch.cat([seg.mean(1), seg.amax(1), glob]))
    if not rows:
        return f.new_zeros(0, 3 * f.shape[0])
    return torch.stack(rows)


@torch.no_grad()
def fit_omni_ridge(
    dataset,      # HTStepDataset of the train split, built with is_training=False (no random crop)
    teacher_mean, # (3584,) mean of the train teacher vectors
    device,
    lam=RIDGE_LAMBDA,
):
    """W = (A^T A + lam I)^-1 A^T (Y - mean), A = the standardised segment inputs of every train segment that has a
    teacher. Returns {'mean', 'std', 'W'} on the device."""
    xs, ys = [], []
    for i in range(len(dataset)):
        x = dataset[i]
        keep = x['omni_ok'][x['labels']]
        if not bool(keep.any()):
            continue
        xs.append(segment_inputs(x['feats']['visual'], x['feats']['audio'], x['segments'][keep]))
        ys.append(x['omni_seg'][x['labels'][keep]].float())
    A = torch.cat(xs).to(device, torch.float64)                    # (S, 4608)
    Y = torch.cat(ys).to(device, torch.float64) - teacher_mean.to(device, torch.float64)
    mean, std = A.mean(0), A.std(0) + 1e-6
    A = (A - mean) / std
    W = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=device, dtype=A.dtype), A.T @ Y)
    print(f'omni ridge: fitted on {len(A)} train segments (lambda {lam:g})', flush=True)
    return {'mean': mean.float(), 'std': std.float(), 'W': W.float()}


def predict_omni_ridge(
    ridge,         # fit_omni_ridge output
    inputs,        # (n, 4608) segment_inputs
    teacher_mean,  # (3584,)
):
    """Normalised omni vectors (n, 3584) predicted by the ridge."""
    x = (inputs.to(ridge['W'].device) - ridge['mean']) / ridge['std']
    return torch.nn.functional.normalize(x @ ridge['W'] + teacher_mean.to(x.device), dim=-1)
