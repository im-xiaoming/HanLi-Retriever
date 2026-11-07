"""Check that the k400 features and the OmniRetriever segment vectors match the annotations, before training.

  python tools/validate_htstep.py                    # datasets/ as laid out by tools/fetch_htstep.py
  python tools/validate_htstep.py --report datasets/validation_report.json

For every train / val_seen video of annotations.json:
  k400  (k400_feats/<id>.npz)
    - the file opens, has k768 and a768, both (T, 768), finite, with the same number of rows T
    - T is close to the duration in the annotations (one row per second)
    - the visual rows are not all zero
  omni  (omni_av_feats/train|val/<id>.npz)
    - the file sits in the folder of its subset (a val video in train/ would leak into training)
    - emb (k, 3584) finite and L2-normalised; seg_idx, step_id, segment (k,) as long as emb
    - every seg_idx points into the video's annotations, with the same step id and the same segment
  both
    - videos with omni vectors but no k400 features (unusable) and train videos with k400 but no omni
      (they train without the omni loss)

Prints a summary; with --report also writes every problem, video by video, to a json file.
Exit code 1 if a blocking problem was found (broken files, mismatched omni rows, val vectors in train/).
"""
import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_k400(path, duration):
    """Problems of one k400 feature file (empty list = fine), and its number of rows."""
    problems = []
    try:
        with np.load(path) as z:
            if 'k768' not in z.files or 'a768' not in z.files:
                return [f'missing keys, has {z.files}'], 0
            video = z['k768'].astype(np.float32)
            audio = z['a768'].astype(np.float32)
    except Exception as e:   # broken zip, truncated download, ...
        return [f'cannot read: {type(e).__name__}: {e}'], 0

    if video.ndim != 2 or video.shape[1] != 768 or audio.ndim != 2 or audio.shape[1] != 768:
        problems.append(f'bad shapes k768 {video.shape} a768 {audio.shape}')
        return problems, 0
    rows = video.shape[0]
    if rows == 0:
        return ['no rows'], 0
    if audio.shape[0] != rows:
        problems.append(f'k768 has {rows} rows, a768 has {audio.shape[0]}')
    if not np.isfinite(video).all() or not np.isfinite(audio).all():
        problems.append('NaN or inf values')
    if np.abs(video).sum() == 0:
        problems.append('visual rows all zero')
    if duration:
        # one row per second: a big gap means a different (or cut) video file
        if abs(rows - duration) > max(5.0, 0.05 * duration):
            problems.append(f'{rows} rows for a {duration:.0f} s video')
    return problems, rows


def check_omni(path, annotations):
    """Problems of one omni file (empty list = fine), and the number of segments it covers."""
    try:
        with np.load(path) as z:
            emb = z['emb'].astype(np.float32)
            seg_idx = z['seg_idx']
            step_id = z['step_id']
            segment = z['segment']
    except Exception as e:
        return [f'cannot read: {type(e).__name__}: {e}'], 0

    problems = []
    k = emb.shape[0]
    if emb.ndim != 2 or emb.shape[1] != 3584:
        return [f'bad emb shape {emb.shape}'], 0
    if not (len(seg_idx) == len(step_id) == len(segment) == k):
        return [f'lengths differ: emb {k}, seg_idx {len(seg_idx)}, step_id {len(step_id)}, segment {len(segment)}'], 0
    if not np.isfinite(emb).all():
        problems.append('NaN or inf values')
    norms = np.linalg.norm(emb, axis=1)
    if k and np.abs(norms - 1).max() > 0.02:
        problems.append(f'not L2-normalised (norms {norms.min():.3f} .. {norms.max():.3f})')
    if len(set(seg_idx.tolist())) != k:
        problems.append('the same seg_idx twice')

    for i in range(k):
        s = int(seg_idx[i])
        if s < 0 or s >= len(annotations):
            problems.append(f'row {i}: seg_idx {s} outside the {len(annotations)} annotations')
            continue
        ann = annotations[s]
        if int(step_id[i]) != int(ann['id']):
            problems.append(f'row {i}: step_id {int(step_id[i])} but annotation {s} is step {ann["id"]}')
        if not np.allclose(segment[i], ann['segment'], atol=1e-3):
            problems.append(f'row {i}: segment {segment[i].tolist()} but annotation {s} is {ann["segment"]}')
    return problems, k


def check_video(job):
    """All the checks of one video. job = (id, subset, duration, annotations, k400 path, omni paths)."""
    vid, subset, duration, annotations, k400_path, omni_paths = job
    result = {'id': vid, 'subset': subset, 'k400': None, 'k400_rows': 0, 'omni': None, 'omni_rows': 0,
              'omni_folder': None, 'n_segments': sum(a['segment'][1] > a['segment'][0] for a in annotations)}
    if os.path.exists(k400_path):
        result['k400'], result['k400_rows'] = check_k400(k400_path, duration)
    for folder, path in omni_paths.items():
        if os.path.exists(path):
            result['omni_folder'] = folder
            result['omni'], result['omni_rows'] = check_omni(path, annotations)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'datasets'))
    ap.add_argument('--json_file', default=None, help='default: <data>/annotations.json')
    ap.add_argument('--report', default='', help='write every problem to this json file')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    db = json.load(open(a.json_file or os.path.join(a.data, 'annotations.json')))
    jobs = []
    for vid, v in db.items():
        subset = v['subset'].lower()
        if subset not in ('train', 'val_seen'):
            continue
        omni_paths = {part: os.path.join(a.data, 'omni_av_feats', part, vid + '.npz') for part in ('train', 'val')}
        jobs.append((vid, subset, v.get('duration'), v['annotations'],
                     os.path.join(a.data, 'k400_feats', vid + '.npz'), omni_paths))

    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(check_video, jobs, chunksize=32))

    # ---- summary ----
    count = defaultdict(int)
    problems = {}
    blocking = 0
    for r in results:
        s = r['subset']
        has_k400 = r['k400'] is not None
        has_omni = r['omni'] is not None
        count[f'{s}: annotated'] += 1
        count[f'{s}: k400 file'] += has_k400
        count[f'{s}: omni file'] += has_omni
        count[f'{s}: k400 + omni'] += has_k400 and has_omni
        video_problems = []
        if has_k400 and r['k400']:
            video_problems += [f'k400: {p}' for p in r['k400']]
            count[f'{s}: k400 with problems'] += 1
            blocking += any(p.startswith(('cannot read', 'missing keys', 'bad shapes', 'no rows', 'NaN')) for p in r['k400'])
        if has_omni:
            expected = 'train' if s == 'train' else 'val'
            if r['omni_folder'] != expected:
                video_problems.append(f'omni: in {r["omni_folder"]}/ but the video is {s}')
                count[f'{s}: omni in the wrong folder'] += 1
                blocking += 1
            if r['omni']:
                video_problems += [f'omni: {p}' for p in r['omni']]
                count[f'{s}: omni with problems'] += 1
                blocking += 1
            if r['omni_rows'] < r['n_segments']:
                count[f'{s}: omni covers only part of the segments'] += 1
            if not has_k400:
                count[f'{s}: omni without k400 (unusable)'] += 1
        elif has_k400 and s == 'train':
            count['train: k400 without omni (no omni loss)'] += 1
        if video_problems:
            problems[r['id']] = video_problems

    for key in sorted(count):
        print(f'{key:50s} {count[key]}')
    print(f'videos with problems: {len(problems)}  (blocking: {blocking})')
    for vid in list(problems)[:15]:
        print(' ', vid, problems[vid][:3])
    if a.report:
        json.dump({'counts': dict(count), 'problems': problems}, open(a.report, 'w'), indent=1)
        print('report:', a.report)
    raise SystemExit(1 if blocking else 0)


if __name__ == '__main__':
    main()
