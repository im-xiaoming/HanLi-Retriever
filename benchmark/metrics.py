"""Validation metrics of the model: article grounding mAP (as the official HT-Step code) and the distance between
the omni head (student) and OmniRetriever (teacher)."""
import numpy as np
import torch


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
    """mAP of HT-Step article grounding, as benchmark/evaluation/eval_article_grounding.py (on the videos given):
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


def omni_student_teacher_scores(
    students,      # list of (n, 3584) student vectors of the GT segments, one entry per video, L2-normalised
    teachers,      # list of (n, 3584) teacher (OmniRetriever) vectors of the same segments, L2-normalised
    teacher_mean,  # (3584,) mean of the train teacher vectors
):
    """Distance between the omni head (student) and OmniRetriever (teacher) on the validation segments.

    OmniCosC: mean cosine similarity between the student and the teacher of the same segment, x100, after
              subtracting the train teacher mean from both vectors: only what is specific to the segment counts
              (0 = no better than the mean). Without the subtraction, predicting the train mean for every segment
              already gives ~80.
    OmniR@1 : share of segments whose own teacher is the closest of all validation teachers to its student
    """
    students = torch.cat(students).float()   # (S, 3584)
    teachers = torch.cat(teachers).float()   # (S, 3584)

    mean = teacher_mean.float().cpu()
    omni_cos_c = float(100 * torch.nn.functional.cosine_similarity(students - mean, teachers - mean, dim=-1).mean())

    # retrieval: compare every student with every teacher, in chunks to keep the memory small
    hits = 0
    for start in range(0, len(students), 1024):
        chunk = students[start:start + 1024]                      # (c, 3584)
        closest = (chunk @ teachers.t()).argmax(dim=-1)           # (c,) index of the closest teacher
        own = torch.arange(start, start + len(chunk))             # (c,) index of its own teacher
        hits += int((closest == own).sum())
    omni_r1 = float(100 * hits / len(students))
    return omni_cos_c, omni_r1
