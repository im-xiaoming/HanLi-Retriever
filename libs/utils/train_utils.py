import os
import copy
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .cider import cider
from .lr_schedulers import LinearWarmupCosineAnnealingLR


def fix_random_seed(seed, include_cuda=True):
    """Seed python, numpy and torch; with include_cuda also make cuDNN / cuBLAS deterministic.
    Returns the torch generator for the DataLoader."""
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if include_cuda:
        # no cudnn benchmark, so a rerun gives the same results
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    return rng_generator


def save_checkpoint(state, file_folder, file_name):
    """write a checkpoint to a file"""
    os.makedirs(file_folder, exist_ok=True)
    torch.save(state, os.path.join(file_folder, file_name))


def make_optimizer(model, optimizer_config):
    """AdamW with three parameter groups (as ActionFormer, plus one for the caption projection):
      decay    : weights of the Linear / Conv layers, with weight decay
      no_decay : biases, LayerNorm, scales and other parameters, no weight decay
      clip     : clip_proj (caption vectors into the event space), its own learning rate clip_proj_lr
    """
    decay, no_decay, clip = [], [], []
    for mod_name, mod in model.named_modules():
        for par_name, p in mod.named_parameters(recurse=False):
            if 'clip_proj' in mod_name:
                clip.append(p)
            elif par_name.endswith('weight') and isinstance(mod, (torch.nn.Linear, torch.nn.Conv1d)):
                decay.append(p)
            else:
                no_decay.append(p)
    groups = [
        {'params': decay, 'weight_decay': optimizer_config['weight_decay']},
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': clip, 'weight_decay': 0.0, 'lr': optimizer_config['clip_proj_lr']},
    ]
    return torch.optim.AdamW(groups, lr=optimizer_config['learning_rate'])


def make_scheduler(optimizer, optimizer_config, num_iters_per_epoch):
    """Linear warm-up over warmup_epochs, then a cosine decay; steps after every iteration."""
    warmup_steps = optimizer_config['warmup_epochs'] * num_iters_per_epoch
    max_steps = (optimizer_config['epochs'] + optimizer_config['warmup_epochs']) * num_iters_per_epoch
    return LinearWarmupCosineAnnealingLR(optimizer, warmup_steps, max_steps)


class ModelEma(torch.nn.Module):
    """Exponential moving average (EMA) of the weights, as ActionFormer: evaluation and checkpoints
    both use these averaged weights."""

    def __init__(self, model, decay=0.999):
        super().__init__()
        # a copy of the model accumulates the moving average
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        # the decay ramps up: with a fixed 0.999 the weights are still almost the initial ones after the
        # first epoch, and the first evaluations would score a model that has learnt nothing
        model = getattr(model, 'module', model)   # the model inside DistributedDataParallel
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(model_v.detach(), alpha=1 - d)
            else:
                ema_v.copy_(model_v)


def train_one_epoch(
    train_loader,      # DataLoader, every batch is a list of dicts (see HTStepDataset.__getitem__)
    model,             # the model, or the DistributedDataParallel around it
    optimizer,
    scheduler,
    model_ema,         # ModelEma, updated after every step
    clip_grad_l2norm,  # maximum L2 norm of the gradients
    device,
):
    """Train one epoch, returns the mean of every loss (over this process's batches)."""
    model.train()
    losses = {}
    for video_list in train_loader:
        for x in video_list:
            x['feats'] = {k: v.to(device) for k, v in x['feats'].items()}
        losses_dict = model(video_list)
        optimizer.zero_grad(set_to_none=True)
        losses_dict['final_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
        optimizer.step()
        scheduler.step()   # the learning rate steps after every iteration, not every epoch
        model_ema.update(model)
        for k, v in losses_dict.items():
            losses[k] = losses.get(k, 0.0) + v.item() / len(train_loader)
    return losses


def iou(a, b):
    """IoU of two segments (start, end)."""
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u > 0 else 0.0


def voc_ap(
    tp,    # (N,) 1 = true positive, predictions sorted by score (best first)
    n_gt,  # number of GT segments
):
    """Interpolated average precision (VOC 2011), as the HT-Step evaluation."""
    tp_c = np.cumsum(tp)
    mprec = np.hstack([[0], tp_c / np.arange(1, len(tp) + 1), [0]])
    mrec = np.hstack([[0], tp_c / n_gt, [1]])
    mprec = np.maximum.accumulate(mprec[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def article_grounding_map(
    preds,  # per video: (video_id, activity, segments (K, 2) in s, scores (K,), steps (K,), steps of its article)
    gts,    # per video: (video_id, activity, GT segments [[s, e], ...], their steps)
    tious=np.linspace(0.3, 0.7, 5),
):
    """mAP of HT-Step article grounding, as data/htstep/evaluation/eval_article_grounding.py (on the videos given):
    predictions whose step is not in the video's article, or is no GT step of any video, are dropped; one AP per
    activity, where a prediction matches any GT segment of its video (the step itself is not compared);
    mean over activities and tIoUs."""
    gt_steps = {s for g in gts for s in g[3]}
    aps = []
    for act in sorted({g[1] for g in gts}):
        gt = {v: np.asarray(s, dtype=float).reshape(-1, 2) for v, a, s, _ in gts if a == act}
        rows = [(v, seg, sc) for v, a, segs, scs, labs, cand in preds if a == act
                for seg, sc, lab in zip(segs.tolist(), scs.tolist(), labs.tolist())
                if lab in cand and lab in gt_steps]
        rows = [rows[i] for i in np.argsort([r[2] for r in rows])[::-1]]
        ap = []
        for thr in tious:
            used = {v: np.zeros(len(g), dtype=bool) for v, g in gt.items()}
            tp = np.zeros(len(rows))
            for i, (v, (s, e), _) in enumerate(rows):
                g = gt[v]
                inter = (np.minimum(e, g[:, 1]) - np.maximum(s, g[:, 0])).clip(0)
                tiou = inter / ((e - s) + (g[:, 1] - g[:, 0]) - inter)
                for j in tiou.argsort()[::-1]:   # the free GT segment with the highest tIoU
                    if tiou[j] < thr:
                        break
                    if not used[v][j]:
                        used[v][j], tp[i] = True, 1
                        break
            ap.append(voc_ap(tp, sum(len(g) for g in gt.values())))
        aps.append(ap)
    return 100 * float(np.mean(aps)) if aps else 0.0


@torch.no_grad()
def valid_one_epoch(
    val_loader,  # DataLoader of the validation split
    model,       # the EMA model
    pool_text,   # [4958] headline of every step, in pool order
    device,
):
    model.eval()
    best_ious, correct = [], []
    gts_txt, hyp = {}, {}
    preds, gt_all = [], []
    art_ok = []
    omni_students, omni_teachers = [], []   # one (n, 3584) tensor per video with omni teacher vectors
    for video_list in val_loader:
        for x in video_list:
            x['feats'] = {k: v.to(device) for k, v in x['feats'].items()}
        for r, x in zip(model(video_list), video_list):
            gts = r['gt_segments']
            # keep as many predictions (best first) as the video has GT steps
            k = min(len(gts), len(r['segments']))
            top = r['segments'][:k].tolist()
            for gi, gt in enumerate(gts):
                vals = [iou(p, gt) for p in top] or [0.0]
                j = int(np.argmax(vals))   # the prediction matching this GT step best
                best_ious.append(vals[j])
                if vals[j] < 0.3:          # captions are only scored on matched steps
                    continue
                step = int(r['labels'][j])
                gts_txt[len(gts_txt)] = [x['cap_text'][gi]]
                hyp[len(hyp)] = [pool_text[step]]
                correct.append(step == x['gt_steps'][gi])
            preds.append((x['video_id'], x['activity'], r['segments'], r['scores'], r['labels'],
                          set(x['cand'].tolist())))
            gt_all.append((x['video_id'], x['activity'], gts, x['gt_steps']))
            art_ok.append(r['article_pred'] == x['article'])
            omni_students.append(r['omni_student'])
            omni_teachers.append(r['omni_teacher'])

    omni_cos, omni_cos_c, omni_r1 = omni_student_teacher_scores(
        omni_students, omni_teachers, model.omni_teacher_mean)
    best_ious = np.array(best_ious)
    return {'R@0.3': float(100 * np.mean(best_ious >= 0.3)),
            'R@0.5': float(100 * np.mean(best_ious >= 0.5)),
            'R@0.7': float(100 * np.mean(best_ious >= 0.7)),
            'mIoU': float(100 * np.mean(best_ious)),
            'CIDEr': cider(gts_txt, hyp) * 100 if hyp else 0.0,
            'Acc': float(100 * np.mean(correct)) if correct else 0.0,
            'mAP': article_grounding_map(preds, gt_all),
            'ArtAcc': float(100 * np.mean(art_ok)),
            'OmniCos': omni_cos,
            'OmniCosC': omni_cos_c,
            'OmniR@1': omni_r1}


def omni_student_teacher_scores(
    students,      # list of (n, 3584) student vectors of the GT segments, one entry per video, L2-normalised
    teachers,      # list of (n, 3584) teacher (OmniRetriever) vectors of the same segments, L2-normalised
    teacher_mean,  # (3584,) mean of the train teacher vectors
):
    """Distance between the omni head (student) and OmniRetriever (teacher) on the validation segments.

    OmniCos : mean cosine similarity between the student and the teacher of the same segment, x100
              (100 = identical direction). Predicting the train mean for every segment already gives ~80.
    OmniCosC: the same after subtracting the train teacher mean from both vectors: only what is specific to the
              segment counts (0 = no better than the mean)
    OmniR@1 : share of segments whose own teacher is the closest of all validation teachers to its student
    """
    students = torch.cat(students).float()   # (S, 3584)
    teachers = torch.cat(teachers).float()   # (S, 3584)

    cosine = torch.nn.functional.cosine_similarity
    omni_cos = float(100 * cosine(students, teachers, dim=-1).mean())

    mean = teacher_mean.float().cpu()
    omni_cos_c = float(100 * cosine(students - mean, teachers - mean, dim=-1).mean())

    # retrieval: compare every student with every teacher, in chunks to keep the memory small
    hits = 0
    for start in range(0, len(students), 1024):
        chunk = students[start:start + 1024]                      # (c, 3584)
        closest = (chunk @ teachers.t()).argmax(dim=-1)           # (c,) index of the closest teacher
        own = torch.arange(start, start + len(chunk))             # (c,) index of its own teacher
        hits += int((closest == own).sum())
    omni_r1 = float(100 * hits / len(students))
    return omni_cos, omni_cos_c, omni_r1
