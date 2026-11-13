"""HT-Step test split features on Colab: fetch inputs, and upload the .npz so far to HF.

  python test_feats_sync.py setup SPLIT      # checkpoints, the split's features already on HF -> feats/SPLIT, its videos not yet extracted -> videos/SPLIT
  python test_feats_sync.py upload           # feats/<split>.tar (npz so far + missing.txt) for every split that changed
  python test_feats_sync.py loop [MIN]       # upload every MIN minutes (default 10)

Videos: nguyenminh04/htstep-test-videos (<split>/<id>.mp4, manifest.json). Features: nguyenminh04/htstep-test-feats,
feats/<split>.tar = the split's <id>.npz extracted so far + missing.txt (ids of the downloaded videos not extracted yet).
Run from /content, with HF_TOKEN set. extract_features.py writes feats/<split>/<id>.npz.
"""
import glob
import json
import os
import sys
import tarfile
import time

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

VIDEOS = 'nguyenminh04/htstep-test-videos'
FEATS = 'nguyenminh04/htstep-test-feats'
SPLITS = ['test_seen', 'test_unseen']
TOKEN = os.environ['HF_TOKEN']
api = HfApi(token=TOKEN)


def downloaded(split):
    man = json.load(open(hf_hub_download(VIDEOS, 'manifest.json', repo_type='dataset', token=TOKEN)))[split]
    return sorted(man['ok'] if isinstance(man, dict) else man)


def done(split, settled=0):
    """ids with an .npz; settled: only files untouched for that many seconds (not being written)."""
    return {os.path.basename(p)[:-4] for p in glob.glob('feats/%s/*.npz' % split)
            if time.time() - os.path.getmtime(p) >= settled}


def setup(split):
    for repo, f in (('OpenGVLab/InternVideo2-Stage1-1B-224p-K400', '1B_ft_k710_ft_k400_f16.pth'),
                    ('OpenGVLab/InternVideo2-Stage2-6B-Audio', 'audio_6b.pth')):
        hf_hub_download(repo, f, local_dir='ckpt/internvideo2', token=TOKEN)
    os.makedirs('feats/' + split, exist_ok=True)
    try:   # features already on HF: extract_features.py skips these videos
        tarfile.open(hf_hub_download(FEATS, 'feats/%s.tar' % split, repo_type='dataset', token=TOKEN,
                                     force_download=True)).extractall('feats/' + split)
    except Exception as e:
        print('no %s tar on HF yet: %r' % (split, e), flush=True)
    if os.path.exists('feats/%s/missing.txt' % split):
        os.remove('feats/%s/missing.txt' % split)
    todo = [v for v in downloaded(split) if v not in done(split)]
    print('%s: %d on HF, %d to extract' % (split, len(done(split)), len(todo)), flush=True)
    snapshot_download(VIDEOS, repo_type='dataset', token=TOKEN, local_dir='videos', max_workers=8,
                      allow_patterns=['%s/%s.mp4' % (split, v) for v in todo])
    open('videos/%s/.complete' % split, 'w').close()
    print('%s: videos ready' % split, flush=True)


def upload(last={}):
    for split in SPLITS:
        ids = done(split, settled=60)
        if not ids or last.get(split) == len(ids):
            continue
        missing = [v for v in downloaded(split) if v not in ids]
        tar = 'feats/.%s.tar' % split
        with tarfile.open(tar, 'w') as t:
            for v in sorted(ids):
                t.add('feats/%s/%s.npz' % (split, v), v + '.npz')
            with open('feats/.missing.txt', 'w') as f:
                f.write('\n'.join(missing) + '\n')
            t.add('feats/.missing.txt', 'missing.txt')
        api.upload_file(path_or_fileobj=tar, path_in_repo='feats/%s.tar' % split, repo_id=FEATS, repo_type='dataset',
                        commit_message='feats/%s.tar: %d videos, %d missing' % (split, len(ids), len(missing)))
        last[split] = len(ids)
        print('%s uploaded %s: %d videos, %d missing' % (time.strftime('%H:%M:%S'), split, len(ids), len(missing)),
              flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'setup':
        setup(sys.argv[2])
    elif cmd == 'upload':
        upload()
    elif cmd == 'loop':
        while True:
            time.sleep(60 * float(sys.argv[2] if len(sys.argv) > 2 else 10))
            try:
                upload()
            except Exception as e:   # a failed upload is retried at the next round
                print('upload failed: %r' % e, flush=True)
