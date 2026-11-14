"""Validation of a model on val_seen: run it on every video, then score its predictions (benchmark/metrics.py)."""
import numpy as np
import torch

from .cider import cider
from .metrics import iou, article_grounding_map, omni_student_teacher_scores


def write_predictions_csv(
    preds,     # per video: (video_id, activity, segments (K, 2) in s, scores (K,), steps (K,) pool rows, cand)
    pool_ids,  # global_step_index of every pool row
    path,
):
    """Predictions in the format of the official HT-Step evaluation (video-id,t-start,t-end,label,score),
    label = global_step_index. Scored by benchmark/eval_htstep.py."""
    with open(path, 'w') as f:
        f.write('video-id,t-start,t-end,label,score\n')
        for video_id, _, segs, scores, steps, _ in preds:
            for (s, e), score, step in zip(segs.tolist(), scores.tolist(), steps.tolist()):
                f.write(f'{video_id},{s:.3f},{e:.3f},{pool_ids[step]},{score:.6f}\n')


@torch.no_grad()
def valid_one_epoch(
    val_loader,  # DataLoader of the validation split
    model,       # the EMA model
    pool_text,   # [4958] headline of every step, in pool order
    device,
    pool_ids=None,   # [4958] global_step_index of every pool row, needed by pred_csv
    pred_csv='',     # if given, the predictions are also written there, in the official HT-Step format
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

    if pred_csv:
        write_predictions_csv(preds, pool_ids, pred_csv)
    omni_cos_c, omni_r1 = omni_student_teacher_scores(omni_students, omni_teachers, model.omni_teacher_mean)
    best_ious = np.array(best_ious)
    return {'R@0.3': float(100 * np.mean(best_ious >= 0.3)),
            'R@0.5': float(100 * np.mean(best_ious >= 0.5)),
            'R@0.7': float(100 * np.mean(best_ious >= 0.7)),
            'mIoU': float(100 * np.mean(best_ious)),
            'CIDEr': cider(gts_txt, hyp) * 100 if hyp else 0.0,
            'Acc': float(100 * np.mean(correct)) if correct else 0.0,
            'mAP': article_grounding_map(preds, gt_all),
            'ArtAcc': float(100 * np.mean(art_ok)),
            'OmniCosC': omni_cos_c,
            'OmniR@1': omni_r1}
