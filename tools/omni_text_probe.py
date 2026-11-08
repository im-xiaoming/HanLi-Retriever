"""Zero-shot probe of the OmniRetriever step text vectors (datasets/step_emb_omni.npz, tools/omni_step_text.py) on the
val_seen GT segments (start and end given), no training:

  step choice among the steps of the video's article (as the article grounding task):
    iv2           : the model's segment vector vs the InternVideo2 step texts (what the model uses now)
    teacher_text  : the OmniRetriever AV vector of the segment (teacher) vs the omni step texts (upper bound)
    student_text  : the model's omni span head vector vs the omni step texts
    iv2 + w * omni: log p(step) of both, added (w on a grid)
  article choice among all 617 articles:
    head          : the model's article head
    teacher_text  : mean teacher AV vector of the video vs the article vectors (mean of their omni step texts)
    student_text  : the same with the span head vectors

    python tools/omni_text_probe.py --ckpt ckpt/pc_v6/model_best.pth.tar
"""
import argparse

import numpy as np
import torch
from torch.nn import functional as F

from libs.core.config import cfg
from libs.datasets import make_dataset
from libs.datasets.helper import load_step_vectors, preprocessing
from libs.modeling import make_multimodal_meta_arch
from libs.modeling.heads import span_mean


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--omni-text', default='./datasets/step_emb_omni.npz')
    ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_cfg = dict(cfg['dataset'], scale_factor=cfg['model']['scale_factor'],
                    regression_range=cfg['model']['regression_range'], backbone_arch=cfg['model']['backbone_arch'])
    val = make_dataset('htstep', False, ['val_seen'], **data_cfg)
    model = make_multimodal_meta_arch('EventCaptionTransformer', **cfg['model']).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)['state_dict'])
    model.eval()
    article_steps = [val.article_steps[x] for x in val.articles]
    model.set_step_pool(torch.from_numpy(val.pool_emb).to(dev), article_steps)

    # omni text vector of every step (pool order) and of every article (mean of its steps, normalised)
    omni_text = torch.from_numpy(load_step_vectors(a.omni_text, val.step_row)).to(dev)            # (4958, 3584)
    omni_text = F.normalize(omni_text, dim=-1)
    omni_article = F.normalize(torch.stack([omni_text[s].mean(0) for s in article_steps]), dim=-1)  # (617, 3584)

    pool_f = model.embed_head.embed_captions(model.pool_raw)                                         # (4958, D)
    scale = model.embed_head.logit_scale.exp().clamp(max=100)

    rows = []       # per GT segment with a teacher: (iv2 logits, teacher logits, student logits, gt column)
    art_all = []    # per val video: (gt article, head logits, whole-video omni scores)
    art = []        # per video: (gt article, head pred, teacher pred, student pred)
    for start in range(0, len(val), a.batch):
        videos = [val[i] for i in range(start, min(start + a.batch, len(val)))]
        V, A, masks = preprocessing(videos, 0, model.max_seq_len, dev)
        fV, fA, msk = model.backbone(V, A, masks)
        feats = [torch.cat((v, x), dim=1) for v, x in zip(fV, fA)]
        art_logits = model.article_head(feats[0], msk[0], model.article_raw)
        cands = [x['cand'].to(dev) for x in videos]
        C = max(len(c) for c in cands)
        idx = torch.stack([F.pad(c, (0, C - len(c)), value=0) for c in cands])
        pad = torch.stack([torch.arange(C, device=dev) >= len(c) for c in cands])
        feats = model.article_cond(feats, msk, model.pool_raw[idx], pad)
        _, _, raw0 = model.embed_head(feats, msk)
        len0 = msk[0].squeeze(1).sum(-1)

        for b, x in enumerate(videos):
            # article on every video: the head, and the span head on the whole video vs the omni article texts
            whole_seg = torch.tensor([[0.0, float(len0[b]) - 1]], device=dev)
            whole_vec = model.omni_span(feats[0][b], int(len0[b]), whole_seg)[0]
            art_all.append((x['article'], art_logits[b].float().cpu(), (whole_vec @ omni_article.t()).cpu()))
            ok = x['omni_ok'][x['labels']]
            if not ok.any():
                continue
            segs = x['segments'][ok].to(dev).float()
            steps = x['step_idx'][x['labels']][ok].to(dev)                                   # (n,) pool rows
            cand = cands[b]
            gt_col = (cand[None, :] == steps[:, None]).float().argmax(1)                      # (n,)

            seg_vec = model.seg_ctx(raw0[b], int(len0[b]), segs)                              # (n, D)
            iv2 = scale * seg_vec @ pool_f[cand].t()
            teacher = x['omni_seg'][x['labels']][ok].to(dev)                                  # (n, 3584)
            student = model.omni_span(feats[0][b], int(len0[b]), segs)                        # (n, 3584)
            rows.append((iv2.cpu(), (teacher @ omni_text[cand].t()).cpu(),
                         (student @ omni_text[cand].t()).cpu(), gt_col.cpu()))
            # no GT segment needed: the span head on the whole video
            whole = torch.tensor([[0.0, float(len0[b]) - 1]], device=dev)
            video_vec = model.omni_span(feats[0][b], int(len0[b]), whole)[0]                  # (3584,)
            art.append((x['article'], art_logits[b].float().cpu(),
                        (F.normalize(teacher.mean(0), dim=0) @ omni_article.t()).cpu(),
                        (F.normalize(student.mean(0), dim=0) @ omni_article.t()).cpu(),
                        (video_vec @ omni_article.t()).cpu()))

    def acc(logits, gt):
        """Share of segments (x100) whose highest score is their GT step."""
        correct = sum(int((l.argmax(1) == g).sum()) for l, g in zip(logits, gt))
        return 100.0 * correct / sum(len(g) for g in gt)

    iv2, tea, stu, gt = zip(*rows)
    n = sum(len(g) for g in gt)
    print(f'{n} GT segments with a teacher, {len(art)} videos')
    print(f'step acc  iv2 {acc(iv2, gt):.2f}   teacher_text {acc(tea, gt):.2f}   student_text {acc(stu, gt):.2f}')
    for temp in (0.02, 0.05):
        for w in (0.25, 0.5, 1.0, 2.0):
            fused = [F.log_softmax(i, 1) + w * F.log_softmax(s / temp, 1) for i, s in zip(iv2, stu)]
            print(f'  iv2 + {w} * student_text (temp {temp}): {acc(fused, gt):.2f}')
    g = np.array([x[0] for x in art])
    def art_acc(scores):
        return 100.0 * float(np.mean([int(sc.argmax()) == gi for sc, gi in zip(scores, g)]))
    head = [x[1] for x in art]
    print(f'article acc  head {art_acc(head):.2f}   teacher_text {art_acc([x[2] for x in art]):.2f}   '
          f'student_text {art_acc([x[3] for x in art]):.2f}')
    for temp in (0.02, 0.05):
        for w in (0.25, 0.5, 1.0):
            fused = [F.log_softmax(x[1], 0) + w * F.log_softmax(x[3] / temp, 0) for x in art]
            whole = [F.log_softmax(x[1], 0) + w * F.log_softmax(x[4] / temp, 0) for x in art]
            print(f'  head + {w} * omni (temp {temp}):  GT segments {art_acc(fused):.2f}   '
                  f'whole video {art_acc(whole):.2f}')
    print(f'  whole video alone: {art_acc([x[4] for x in art]):.2f}')

    ga = np.array([x[0] for x in art_all])
    def acc_all(scores):
        return 100.0 * float(np.mean([int(sc.argmax()) == gi for sc, gi in zip(scores, ga)]))
    print(f'ALL {len(art_all)} val videos: head {acc_all([x[1] for x in art_all]):.2f}   '
          f'whole-video omni alone {acc_all([x[2] for x in art_all]):.2f}')
    for temp in (0.02, 0.05):
        for w in (0.25, 0.5, 1.0, 2.0):
            fused = [F.log_softmax(x[1], 0) + w * F.log_softmax(x[2] / temp, 0) for x in art_all]
            print(f'  head + {w} * whole-video omni (temp {temp}): {acc_all(fused):.2f}')


if __name__ == '__main__':
    main()
