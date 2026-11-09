"""HT-Step metrics of a predictions CSV (written by `python train.py --eval CKPT`), computed as other works do,
to compare with their numbers:

  1. article grounding mAP at tIoU 0.3 / 0.5 / 0.7 and averaged over 0.3:0.1:0.7
     (HT-Step paper, Table 2): the official evaluation code, tools/htstep_eval/eval_article_grounding.py,
     copied unchanged from github.com/facebookresearch/htstep;
  2. the same mAP on full-match and on partial-match steps (HT-Step paper, Table 4 and Appendix D.2):
     a step query (video, step) is "full" if all its GT segments are Full Match, "partial" if all are Partial
     Match, "mixed" otherwise. The full mAP keeps the GT segments of the full queries and drops the predictions
     of the partial and mixed queries (so predictions of steps not in the video still count as false positives);
     the partial mAP the other way round. Activities without GT then get no AP, as in the paper;
  3. R@1 of step grounding (VINA, MPTVA, NaSVA): for every step that occurs in a video, the most relevant
     moment is the centre of the highest-scored predicted segment of that step; the step is recalled if that
     moment lies inside one of its GT segments. R@1 = recalled / occurring (video, step) pairs, pooled.

Every metric is given on the whole val_seen split (600 videos: a video without features has no prediction,
so its steps are missed, as in the official evaluation) and on the videos that have predictions.

    python tools/eval_htstep.py ckpt/htstep/model_best_val_seen.csv
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'htstep_eval'))
from eval_article_grounding import ANETdetectionGrouped, evaluate_article_grounding  # noqa: E402

TIOUS = {'mAP@0.3': [0.3], 'mAP@0.5': [0.5], 'mAP@0.7': [0.7], 'mAP': np.linspace(0.3, 0.7, 5)}


def grounding_map(ants, gt_ants, preds, taxonomy, split, tious):
    """evaluate_article_grounding of the official code, with the GT taken from gt_ants (a filtered copy of
    ants). The predictions are kept as the official code does: only the steps of the video's article
    (activity + variation, read from the full ants)."""
    step_var = dict(zip(taxonomy.global_step_index, taxonomy.variation))
    act_var_steps = {(a, v): set(g.global_step_index) for (a, v), g in taxonomy.groupby(['activity', 'variation'])}
    article = {vid: (x['activity'], step_var[x['annotations'][0]['id']]) for vid, x in ants.items()}
    keep = [article[v] in act_var_steps and lab in act_var_steps[article[v]]
            for v, lab in zip(preds['video-id'], preds['label'])]

    act_steps = {a: g.global_step_index.values for a, g in taxonomy.groupby('activity')}
    act_idx = {a: i for i, (a, s) in enumerate(sorted(act_steps.items(), key=lambda x: x[1][0]))}
    step_act = {s: act_idx[a] for a, steps in act_steps.items() for s in steps}
    evaluator = ANETdetectionGrouped(json.loads(json.dumps(gt_ants)), split, tious, label_groups=step_act, label='id')
    with contextlib.redirect_stdout(io.StringIO()):   # the official code prints its own report
        _, average = evaluator.evaluate(preds[keep].reset_index(drop=True), verbose=False)
    return 100 * float(average)


def query_types(ants):
    """{(video, step): 'full' | 'partial' | 'mixed'} for every step that occurs in a video."""
    kinds = {}
    for vid, x in ants.items():
        for a in x['annotations']:
            kinds.setdefault((vid, a['id']), set()).add(a['partial'])
    return {q: 'full' if k == {'Full Match'} else 'partial' if k == {'Partial Match'} else 'mixed'
            for q, k in kinds.items()}


def keep_queries(ants, preds, kind):
    """GT segments and predictions of the queries of one kind, as Appendix D.2 of the HT-Step paper."""
    types = query_types(ants)
    gt = {vid: dict(x, annotations=[a for a in x['annotations'] if types[(vid, a['id'])] == kind])
          for vid, x in ants.items()}
    other = [types.get((v, lab), kind) != kind for v, lab in zip(preds['video-id'], preds['label'])]
    return gt, preds[~np.array(other, dtype=bool)].reset_index(drop=True)


def recall_at_1(ants, preds):
    """R@1 of step grounding, pooled over the (video, step) pairs that occur, in %, and their number."""
    gt = {}
    for vid, x in ants.items():
        for a in x['annotations']:
            gt.setdefault((vid, a['id']), []).append(a['segment'])
    best = preds.sort_values('score', ascending=False).drop_duplicates(['video-id', 'label'])
    moment = {(v, lab): (s + e) / 2 for v, lab, s, e in zip(best['video-id'], best['label'], best['t-start'], best['t-end'])}
    hits = [q in moment and any(s <= moment[q] <= e for s, e in segs) for q, segs in gt.items()]
    return 100 * float(np.mean(hits)), len(hits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('predictions_csv')
    p.add_argument('--annotations', default='datasets/annotations.json')
    p.add_argument('--taxonomy', default='datasets/taxonomy.csv')
    p.add_argument('--split', default='val_seen')
    a = p.parse_args()

    taxonomy = pd.read_csv(a.taxonomy)
    ants = {k: v for k, v in json.load(open(a.annotations)).items() if v['subset'] == a.split}
    preds = pd.read_csv(a.predictions_csv)
    with_preds = {k: v for k, v in ants.items() if k in set(preds['video-id'])}

    # check: with the full GT, grounding_map gives the number of the official entry point
    with contextlib.redirect_stdout(io.StringIO()):
        official = 100 * evaluate_article_grounding(json.loads(json.dumps(ants)), preds, taxonomy, a.split,
                                                    TIOUS['mAP'])
    mine = grounding_map(ants, ants, preds, taxonomy, a.split, TIOUS['mAP'])
    assert abs(official - mine) < 1e-6, (official, mine)

    for name, videos in [(f'all {a.split} ({len(ants)} videos)', ants),
                         (f'videos with predictions ({len(with_preds)})', with_preds)]:
        print(f'== {name}')
        line = [f'{k} {grounding_map(videos, videos, preds, taxonomy, a.split, t):.2f}' for k, t in TIOUS.items()]
        print('  article grounding: ' + '  '.join(line))
        for kind in ['full', 'partial']:
            gt, kept = keep_queries(videos, preds, kind)
            n_gt = sum(len(x['annotations']) for x in gt.values())
            n_act = len({x['activity'] for x in gt.values() if x['annotations']})
            print(f'  {kind:7s} match mAP {grounding_map(videos, gt, kept, taxonomy, a.split, TIOUS["mAP"]):.2f}'
                  f'  ({n_act} activities, {n_gt} GT segments)')
        r1, n = recall_at_1(videos, preds)
        print(f'  step grounding R@1 {r1:.2f}  ({n} (video, step) pairs)')


if __name__ == '__main__':
    main()
