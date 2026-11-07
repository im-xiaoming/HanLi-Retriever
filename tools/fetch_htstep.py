"""Fetch the HT-Step training data from HF into datasets/ (PC or Colab).

  python tools/fetch_htstep.py                 # k400 features (train + val_seen) and OmniRetriever segment vectors
  python tools/fetch_htstep.py --what omni     # only the omni vectors

Layout written:
  <out>/k400_feats/<id>.npz              k768 + a768, one row per second (htstep-feats-k400, htstep-val-seen-feats)
  <out>/omni_av_feats/train/<id>.npz     OmniRetriever encode_av vector of every GT segment of a train video
  <out>/omni_av_feats/val/<id>.npz       same for the val_seen videos (htstep-val-seen-feats, omni_av/)

Feature tars are unpacked in name order, so the zz_fix_* tars (re-extracted broken videos) override older copies.
Empty npz members are skipped. Needs huggingface_hub; HF_TOKEN in the environment or in .env.
"""
import argparse
import json
import os
import shutil
import tarfile

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K400_REPOS = ['nguyenminh04/htstep-feats-k400', 'nguyenminh04/htstep-val-seen-feats']
OMNI_TRAIN_REPO = 'nguyenminh04/htstep-omni-av'          # feats/<id>.npz, train videos
OMNI_VAL_REPO = 'nguyenminh04/htstep-val-seen-feats'     # omni_av/<id>.npz, val_seen videos


def hf_token():
    if os.environ.get('HF_TOKEN'):
        return os.environ['HF_TOKEN']
    env = os.path.join(ROOT, '.env')
    if os.path.exists(env):
        for line in open(env):
            if line.startswith('HF_TOKEN='):
                return line.split('=', 1)[1].strip()
    return None


def fetch_k400(out, cache, token):
    dst = os.path.join(out, 'k400_feats')
    os.makedirs(dst, exist_ok=True)
    api = HfApi(token=token)
    tars = []
    for repo in K400_REPOS:
        names = [f for f in api.list_repo_files(repo, repo_type='dataset') if f.startswith('feats/') and f.endswith('.tar')]
        tars += [(os.path.basename(f), repo, f) for f in names]
    tars.sort()   # zz_fix_* last: they override
    for i, (name, repo, f) in enumerate(tars):
        p = hf_hub_download(repo, f, repo_type='dataset', token=token, cache_dir=cache)
        n = 0
        with tarfile.open(p) as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith('.npz') or m.size == 0:
                    continue
                with tf.extractfile(m) as src, open(os.path.join(dst, os.path.basename(m.name)), 'wb') as d:
                    shutil.copyfileobj(src, d)
                n += 1
        print(f'[{i + 1}/{len(tars)}] {repo.split("/")[1]}/{f}: {n} npz', flush=True)
        if cache is None:
            continue
        os.remove(os.path.realpath(p))   # keep the disk small: the tar is not needed once unpacked


def fetch_omni(out, cache, token, json_file):
    subset = {k: v['subset'].lower() for k, v in json.load(open(json_file)).items()}
    tmp = os.path.join(out, '_omni_dl')
    for repo, pattern in [(OMNI_TRAIN_REPO, 'feats/*.npz'), (OMNI_VAL_REPO, 'omni_av/*.npz')]:
        snapshot_download(repo, repo_type='dataset', token=token, allow_patterns=[pattern], local_dir=tmp,
                          max_workers=16)
    n = {'train': 0, 'val': 0}
    for d in ['feats', 'omni_av']:
        src = os.path.join(tmp, d)
        if not os.path.isdir(src):
            continue
        for f in os.listdir(src):
            s = subset.get(f[:-4])
            if s is None:
                continue
            part = 'train' if s == 'train' else 'val'
            os.makedirs(os.path.join(out, 'omni_av_feats', part), exist_ok=True)
            shutil.copyfile(os.path.join(src, f), os.path.join(out, 'omni_av_feats', part, f))
            n[part] += 1
    shutil.rmtree(tmp, ignore_errors=True)
    print('omni:', n, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'datasets'))
    ap.add_argument('--what', nargs='+', default=['k400', 'omni'], choices=['k400', 'omni'])
    ap.add_argument('--cache', default=None, help='HF cache dir; tars are deleted from it after unpacking')
    ap.add_argument('--json_file', default=os.path.join(ROOT, 'datasets', 'annotations.json'))
    a = ap.parse_args()
    token = hf_token()
    if 'k400' in a.what:
        fetch_k400(a.out, a.cache, token)
    if 'omni' in a.what:
        fetch_omni(a.out, a.cache, token, a.json_file)


if __name__ == '__main__':
    main()
