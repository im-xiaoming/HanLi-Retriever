# HT-Step article grounding (UniAV + OmniRetriever distillation)

Given an instructional video and its article (the steps of a task), the model finds when each step happens. It also
predicts the article (617 articles) and an OmniRetriever-like vector (3584-d) for every segment.

Backbone: [UniAV](https://arxiv.org/abs/2404.03179) (video: InternVideo2 K400 features, audio: BEATs, cross-attention,
6-level pyramid). Heads: article conditioning, event + boundary, step choice, article head, omni distillation (KL +
cosine to the OmniRetriever `encode_av` vector of each segment).

## Results (val_seen)

Model metrics (510 val_seen videos with features, Colab A100 run):

| mAP (article given) | ArtAcc | OmniCos | OmniCosC | OmniR@1 |
|---|---|---|---|---|
| 30.37 | 59.8 | 91.1 | 75.2 | 60.8 |

Metrics of other works (`tools/eval_htstep.py`, checkpoint of the RTX 3060 run on HF
`nguyenminh04/uniav-htstep-omni`). "600" = the whole val_seen split, the 90 videos without features count as
missed, as in the official evaluation; "510" = the videos with features.

| Videos | mAP@0.3 | mAP@0.5 | mAP@0.7 | mAP@[0.3:0.7] | mAP full match | mAP partial match | R@1 |
|---|---|---|---|---|---|---|---|
| 600 | 37.23 | 25.86 | 13.50 | 25.72 | 28.92 | 21.45 | 36.57 |
| 510 | 42.85 | 29.84 | 15.74 | 29.70 | 34.18 | 25.13 | 42.13 |

- mAP: official HT-Step code (`tools/htstep_eval/`, unchanged). HT-Step paper, Table 2 (seen *test* S1):
  ActionFormer-T 36.9 / 26.5 / 15.3 / 26.3, with HowTo100M pre-training 41.2 / 30.8 / 18.3 / 30.2.
- Full / partial match: as Appendix D.2 of the HT-Step paper; Table 4 (test S1): ActionFormer 29.6 / 20.4.
- R@1: as VINA / MPTVA (the step is recalled if its best moment lies in one of its GT segments, pooled over the
  occurring steps). val_seen is the "HT-Step val" of VINA (600 videos, 3441 segments): VINA 34.0-39.1 on val
  (Table 4); on test VINA 37.4, MPTVA 43.3. These works are weakly supervised (narrations), ours is supervised.

Details: [docs/htstep_report.md](docs/htstep_report.md), every run: [docs/experiments_report.md](docs/experiments_report.md).

## Features

| Input | Model | Settings | Output (HF) |
|---|---|---|---|
| Video `k768` | InternVideo2-Stage1-1B K710→K400 f16 (`1B_ft_k710_ft_k400_f16.pth`) | 4 fps, 16-frame window (~4 s, second *i* = frames 4i−7..4i+8), 224×224 squashed, pooled vector before the K400 classifier | (n_sec, 768) fp16, 1 row/s: `htstep-feats-k400`, `htstep-val-seen-feats` |
| Audio `a768` | BEATs of InternVideo2-Stage2-6B-Audio (`audio_6b.pth`) | 16 kHz mono, 3 s around each second, zeros if no audio | (n_sec, 768) fp16, same files |
| Step text | InternVideo2-Stage2-1B f4 text encoder (BERT-large, first 19 layers, [CLS] → `text_proj`) | `"<activity>: <headline>"` for the 4958 steps, centred on their mean | (4958, 512): `htstep-text-feats/text_feats/step_emb_iv2j.npz` |
| Omni AV teacher | OmniRetriever `encode_av`: WAVE-7B (`nguyenminh04/omni-model`) + HT-Step LoRA (`nguyenminh04/omni-adapters`), bf16 | one vector per GT segment [start, end]: 8 frames uniform (224 px), audio of the same window (16 kHz, 8 s, centre crop / zero pad), prompt "Please describe the video." | (n_seg, 3584) fp16, L2-normalised: `htstep-omni-av`, `htstep-val-seen-feats/omni_av/` |

Scripts:
- video + audio: `tools/pipeline/extract_features.py` (default `--model k400_f16`);
- step text: `python tools/htstep_data.py build`;
- omni AV: `scripts/extract_htstep_av.py` of [Omni-Retriever-Custom](https://github.com/im-xiaoming/Omni-Retriever-Custom) (branch `htstep-av-extract`).

## Run

1. Install (Python 3.13, PyTorch 2.14):

       uv venv -p 3.13 .venv && source .venv/bin/activate
       uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130 \
           --extra-index-url https://pypi.org/simple

2. Data into `datasets/` (needs `HF_TOKEN` in the environment or `.env`):
   - `annotations.json`, `taxonomy.csv` from the [HT-Step repo](https://github.com/facebookresearch/htstep);
   - `step_emb_iv2j.npz` from HF `nguyenminh04/htstep-text-feats`;
   - features and omni vectors: `python tools/fetch_htstep.py`, then check them with `python tools/validate_htstep.py`.

3. Train / evaluate:

       python train.py                                   # one GPU, checkpoints in ckpt/htstep/
       torchrun --nproc_per_node=4 train.py              # several GPUs
       python train.py --output NAME --set opt.epochs=8  # other output folder, override a parameter
       python train.py --eval ckpt/htstep/model_best.pth.tar   # also writes model_best_val_seen.csv
       python tools/eval_htstep.py ckpt/htstep/model_best_val_seen.csv   # metrics of other works

   On Colab (from `/content`): `BRANCH=main bash colab_train.sh setup` (code + data, ~10 min), then
   `bash colab_train.sh train NAME`.

All parameters are in `libs/core/config.py`. Training is deterministic (same code, data and seed give the same
numbers); about 12 epochs, ~45 min on an A100.

## Code

```
train.py              training / evaluation
libs/core/config.py   all parameters
libs/datasets/        HT-Step dataset, file readers, crop, training targets
libs/modeling/        backbone, heads, losses, model (event_archs.py)
libs/utils/           optimizer, EMA, train / valid loop, NMS, metrics
tools/                data preparation (features, text vectors), eval_htstep.py (metrics of other works);
                      htstep_eval/ = official HT-Step evaluation; pipeline/ = video download + feature extraction
```
