# HT-Step step detection and captioning (minimal)

Finds the steps of an instructional video and classifies each one among the 4958 steps of the HT-Step taxonomy; the
caption of a segment is the headline of its step. The model is the [UniAV](https://arxiv.org/abs/2404.03179) backbone
(one ConvTransformer stream for video, one for audio, with cross-attention, 6-level pyramid) with an event score, the
step boundaries and the language-aware classifier of UniAV: the segment vector is compared with the text vectors of
the steps. As in [OV-AVEL](https://arxiv.org/abs/2411.11278) a class is only its text, so any list of steps can be
scored at inference (by default the steps of the video's article, as the HT-Step article grounding task).

Python 3.13 and PyTorch 2.14, the versions of Google Colab. One or several GPUs (DistributedDataParallel).

## Folders

```
train.py                     training / evaluation entry point (one GPU, or torchrun on several)
libs/
  core/config.py             every parameter: paths, model sizes, loss weights, optimisation
  datasets/                  data
    datasets.py              registry: make_dataset, make_data_loader, make_generator (as UniAV)
    htstep.py                HT-Step dataset: features resampled to 256 steps, GT steps, article, omni teachers
    helper.py                file readers (features, step vectors, omni vectors) and batch padding
    data_utils.py            random crop for training, training targets of every pyramid point
    loc_generators.py        the points of the pyramid
  modeling/                  model
    blocks.py                UniAV blocks: MaskedConv1D, LayerNorm, attention, Transformer block
    multimodal_backbones.py  UniAV backbone: video and audio streams, cross-attention, 6-level pyramid
    heads.py                 event, boundary, embedding, article conditioning, article head, omni span head
    losses.py                focal loss, DIoU loss, step InfoNCE, omni distillation (KL + cosine)
    event_archs.py           the model: forward, losses, inference
    models.py                registry of backbones and models (as UniAV)
  utils/                     training helpers
    train_utils.py           seed, optimizer, EMA, one training epoch, validation (mAP, ArtAcc, Omni*, ...)
    nms.py                   1D soft-NMS
    cider.py                 CIDEr metric (same numbers as pycocoevalcap)
    lr_schedulers.py         linear warm-up then cosine decay
    run_record.py            run record: experiments/runs/<host>-<time>.json
tools/                       data preparation, not needed for training once datasets/ is filled
  fetch_htstep.py            k400 features (train, val_seen) and OmniRetriever segment vectors -> datasets/
  validate_htstep.py         checks the features and omni vectors against the annotations
  colab_train.sh             Colab: setup (code, data, validation), then train
  htstep_data.py             build: InternVideo2 step text vectors (step_emb_iv2j.npz)
  iv2_text.py                InternVideo2 text encoder (text -> 512-d vector), used by htstep_data.py build
  omni_step_text.py          OmniRetriever text vectors of the steps (tried, not used by the model)
  pipeline/                  video download + feature extraction for HT-Step, see "Feature pipeline" below
datasets/                    (not in git) annotations.json, taxonomy.csv, step_emb_iv2j.npz,
                             k400_feats/<video_id>.npz, omni_av_feats/{train,val}/<video_id>.npz
ckpt/                        (not in git) checkpoints
experiments/runs/            (created by train.py) one JSON record per training run
```

## Setup

    uv venv -p 3.13 .venv
    VIRTUAL_ENV=.venv uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple
    source .venv/bin/activate

On Colab the default runtime already has Python 3.13 and PyTorch; only `pip install huggingface_hub` is needed.

## Data

- `datasets/annotations.json`, `datasets/taxonomy.csv`: the official [HT-Step repo](https://github.com/facebookresearch/htstep).
- `datasets/step_emb_iv2j.npz`: InternVideo2 text vectors of `"<activity>: <headline>"` for the 4958 steps
  (`python tools/htstep_data.py build`, on HF `nguyenminh04/htstep-text-feats`).
- `python tools/fetch_htstep.py` downloads the k400_f16 features of the annotated videos (HF
  `nguyenminh04/htstep-feats-k400` for train, `nguyenminh04/htstep-val-seen-feats` for val_seen) into
  `datasets/k400_feats/`: one file per video, `k768` (InternVideo2-1B K400 f16, 4 fps, ~4 s window) and `a768`
  (BEATs audio, zeros if the video has no audio), one row per second. It also downloads the OmniRetriever
  `encode_av` vector of every GT segment (`nguyenminh04/htstep-omni-av` for train, `htstep-val-seen-feats/omni_av/`
  for val_seen) into `datasets/omni_av_feats/{train,val}/`. Needs `HF_TOKEN` in the environment or `.env`.
- `python tools/validate_htstep.py` checks them against the annotations; `tools/colab_train.sh` does all of this
  on Colab, then trains.

## Model

The UniAV backbone gives a 6-level pyramid of video + audio features. On top of it:

- **Article conditioning**: the features attend to the step texts of the video's article (given, as in the HT-Step
  article grounding task).
- **Event and boundary heads**: an event score and the distances to the step start and end at every point, plus an
  IoU-quality score used for ranking.
- **Step choice**: the segment vector (mean of the level-0 features over the segment, refined by `SegmentContext`)
  is compared with the step texts of the article (InfoNCE, also on every point inside a GT segment).
- **Article head**: picks the video's article among the 617 (text of its steps + a learned vector per article),
  so the article needs not be given (`test_cfg.article_soft=True`: every step competes with
  `log p(step | segment) + article_soft_weight * log p(its article | video)`).
- **Omni distillation**: the model learns the OmniRetriever `encode_av` vector (teacher, 3584-d) of every GT segment,
  with a point head (every point inside the segment) and a span head (attention pooling of the segment, start and
  end given). Loss: KL distillation of the teacher's similarity distribution over the teachers of the batch, plus
  `1 - cos(student, teacher)`.

Removed after ablations (see docs/experiments_report.md): a mined bank of omni negatives, the same-step mask,
two other backbones (TemporalMaxer, DyFADet), the OmniRetriever step texts. The emb loss, IoU branch,
SegmentContext, EMA and random crop look useless one at a time but cost 1.3-2 mAP when removed together.

## Train

    python train.py                               # one GPU
    torchrun --nproc_per_node=4 train.py          # 4 GPUs: one process each, batch_size videos per GPU
    python train.py --eval ckpt/htstep/model_best.pth.tar   # score a checkpoint

Every epoch prints, on val_seen: R@0.5, mIoU, Acc (matched steps given the right step), mAP (official article
grounding mAP, tIoU 0.3-0.7, same numbers as the official evaluation script on the videos that have features),
ArtAcc (article head), OmniCos (cosine between the span head and the teacher on the GT segments), OmniCosC (the same
after removing the train teacher mean) and OmniR@1 (own teacher is the nearest of all val teachers). The checkpoint
with the best mAP goes to `ckpt/<run>/model_best.pth.tar`, the best ArtAcc to `model_best_article.pth.tar`.
Training stops when none of mAP, ArtAcc and OmniCos has improved by at least `--min_delta` (0.3) for `--patience`
epochs (default 2). Parameters are edited in `libs/core/config.py`, or overridden with
`--set opt.epochs=8 model.train_cfg.loss_weight_span=0.5`; `--output NAME` writes the checkpoints to `ckpt/NAME`.
Training is deterministic: the same code, data and seed give the same numbers.

## Feature pipeline (`tools/pipeline/`)

Downloads the HT-Step videos, extracts the k400_f16 features and uploads one tar of features per chunk
(100 videos) to `nguyenminh04/htstep-feats-k400`. Each machine runs on its own CSV of chunks.

Python, one step each:

| File | What it does |
|---|---|
| `download_hf.py` | downloads the video shards of the CSV's chunks from HF (`nguyenminh04/htstep-videos`) |
| `download_videos.py` | downloads videos from YouTube with yt-dlp (chunks without a video shard) |
| `extract_features.py` | extracts `k768` + `a768` for every downloaded chunk; `--upload` sends each chunk's tar to HF |
| `upload_partial.py` | uploads the features of an unfinished chunk as a partial tar (with the list of missing videos) |

Shell, one per machine:

| File | Where it runs | What it does |
|---|---|---|
| `run.sh CSV` | any machine | starts the three Python steps above in the background for one CSV |
| `colab_dl_launch.sh` | PC | gets a Colab GPU, uploads the pipeline and starts `colab_dl_extract.sh` there |
| `colab_dl_extract.sh` | Colab | installs the dependencies and weights, then `run.sh rest_colab.csv` |
| `colab_keepalive.sh` | PC | keeps the Colab session alive (Colab closes idle CLI sessions) |
| `colab_watchdog.sh` | PC | every 5 min, relaunches `colab_dl_launch.sh` if the Colab extraction died |
| `kaggle_work.sh` | Kaggle | one share per T4 GPU (`shards_part1_kaggle0/1.csv`) |

CSV, one row per chunk (same columns everywhere):

| File | Content |
|---|---|
| `shards.csv` | all 205 chunks |
| `shards_part1.csv`, `shards_part2.csv` | the two halves; part 1 is done on our machines |
| `shards_part1_{pc,colab,kaggle0,kaggle1}.csv` | first split of part 1 between the machines |
| `rest_{pc,colab,vast,kaggle}.csv` | current split of what is left of part 1 |

## Results

The defaults in `libs/core/config.py` are run a_kd (Colab A100, val_seen): mAP 30.37 with the given article,
ArtAcc 59.8, OmniCos 91.1, OmniCosC 75.2, OmniR@1 60.8. All runs: docs/experiments_report.md.
