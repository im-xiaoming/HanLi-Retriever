"""OmniRetriever text vector of every HT-Step step: encode_text("<activity>: <headline>"), the step text format
OmniRetriever was fine-tuned with (scripts/convert_htstep.py), with the same adapter (omniretriever-7b) and the
same defaults as the AV vectors of datasets/omni_av_feats, so both live in one space.

Runs on a GPU with the OmniRetriever code (bf16, default loader):

    python tools/omni_step_text.py --omni-src <omniretriever>/src --base-model $WAVE_HOME/WAVE-7B \
        --adapter <adapters>/omniretriever-7b --taxonomy datasets/taxonomy.csv --out step_emb_omni.npz

Output npz, the same layout as datasets/step_emb_iv2j.npz:
    ids  (4958,)        global_step_index
    emb  (4958, 3584)   float16, L2-normalised
    text (4958,)        the encoded strings
"""
import argparse
import csv
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--omni-src', required=True, help='src folder of the OmniRetriever code')
    ap.add_argument('--base-model', required=True, help='WAVE-7B folder')
    ap.add_argument('--adapter', required=True, help='omniretriever-7b adapter folder')
    ap.add_argument('--taxonomy', required=True, help='HT-Step taxonomy.csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=64)
    a = ap.parse_args()

    sys.path.insert(0, a.omni_src)
    from omniretriever import OmniRetriever

    with open(a.taxonomy, newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r['global_step_index']))
    ids = np.array([int(r['global_step_index']) for r in rows], dtype=np.int64)
    texts = [f"{r['activity']}: {r['headline'].strip()}" for r in rows]
    print(f'{len(texts)} steps, e.g. {texts[0]!r}', flush=True)

    model = OmniRetriever.from_pretrained(a.base_model, a.adapter, device='cuda', dtype='bfloat16')
    vectors = []
    start = time.time()
    for i in range(0, len(texts), a.batch):
        v = model.encode_text(texts[i:i + a.batch]).float().cpu().numpy()
        vectors.append(v)
        print(f'{i + len(v)}/{len(texts)}  {time.time() - start:.0f}s', flush=True)
    emb = np.concatenate(vectors)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    # a batch must give the same vectors as one text at a time (padding must not leak into the vector)
    single = np.concatenate([model.encode_text(t).float().cpu().numpy() for t in texts[:3]])
    single = single / np.linalg.norm(single, axis=1, keepdims=True)
    print('batch vs single cosine:', np.round((single * emb[:3]).sum(1), 4), flush=True)

    np.savez(a.out, ids=ids, emb=emb.astype(np.float16), text=np.array(texts))
    print('saved', a.out, emb.shape, flush=True)


if __name__ == '__main__':
    main()
