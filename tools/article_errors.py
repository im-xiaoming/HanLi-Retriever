"""Where does the article prediction go wrong? Scores a checkpoint on val_seen and splits the article errors.

  python tools/article_errors.py ckpt/<run>/model_best_article.pth.tar [--set model.omni_head_dim=0 ...]

An article is (activity, variation). For every article score (head, step, omni, fused) it prints:
  acc        : right article
  activity   : right activity (dish), whatever the variation
  top5       : right article among the 5 best
  same-dish errors: share of the wrong answers that only got the variation wrong
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.core import cfg                                  # noqa: E402
from libs.datasets import make_dataset, make_data_loader   # noqa: E402
from libs.modeling import make_multimodal_meta_arch         # noqa: E402
from train import apply_overrides                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--set', nargs='+', action='extend', default=[])
    a = ap.parse_args()
    apply_overrides(['model.article_head=True'] + a.set)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_cfg = dict(cfg['dataset'], scale_factor=cfg['model']['scale_factor'],
                    regression_range=cfg['model']['regression_range'], backbone_arch=cfg['model']['backbone_arch'])
    train_set = make_dataset('htstep', True, ['train'], **data_cfg)
    val_set = make_dataset('htstep', False, ['val_seen'], **data_cfg)
    loader = make_data_loader(val_set, False, None, 16, 4, torch.utils.data.SequentialSampler(val_set))

    model = make_multimodal_meta_arch('EventCaptionTransformer', **cfg['model']).to(device)
    model.set_step_pool(torch.from_numpy(train_set.pool_emb).to(device),
                        [train_set.article_steps[x] for x in train_set.articles])
    if model.use_omni:
        vectors, articles = train_set.omni_teacher_bank()
        model.set_omni_teachers(torch.from_numpy(vectors).to(device), torch.from_numpy(articles).to(device))
    model.load_state_dict(torch.load(a.ckpt, map_location=device)['state_dict'])
    model.eval()

    activity_of = np.array([art[0] for art in train_set.articles])   # activity of every article
    rows = {}   # score name -> list of (scores (A,), true article)
    with torch.no_grad():
        for video_list in loader:
            for x in video_list:
                x['feats'] = {k: v.to(device) for k, v in x['feats'].items()}
            for r, x in zip(model(video_list), video_list):
                scores = dict(r['article_scores'])
                fused = 0.0
                for s in scores.values():
                    fused = fused + (s - s.mean()) / s.std().clamp(min=1e-6)
                scores['fused'] = fused
                for name, s in scores.items():
                    rows.setdefault(name, []).append((s.numpy(), x['article']))

    print(f'{"score":8s} {"acc":>6s} {"activity":>9s} {"top5":>6s} {"same-dish errors":>17s}   ({len(rows["head"])} videos)')
    for name, items in rows.items():
        acc = act = top5 = wrong = same_dish = 0
        for s, truth in items:
            pred = int(s.argmax())
            acc += pred == truth
            act += activity_of[pred] == activity_of[truth]
            top5 += truth in np.argsort(-s)[:5]
            if pred != truth:
                wrong += 1
                same_dish += activity_of[pred] == activity_of[truth]
        n = len(items)
        print(f'{name:8s} {100 * acc / n:6.2f} {100 * act / n:9.2f} {100 * top5 / n:6.2f} '
              f'{100 * same_dish / max(wrong, 1):16.1f}%')


if __name__ == '__main__':
    main()
