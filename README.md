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
    htstep.py                HT-Step dataset: features resampled to 256 steps, GT steps, captions
    data_utils.py            random crop for training, training targets of every pyramid point
    loc_generators.py        the points of the pyramid
  modeling/                  model
    blocks.py                UniAV blocks: MaskedConv1D, LayerNorm, attention, Transformer block
    multimodal_backbones.py  UniAV backbone: video and audio streams, cross-attention, 6-level pyramid
    losses.py                focal loss, DIoU loss
    event_archs.py           the heads (event, boundary, step classifier, optional omni / audio), losses, inference
    models.py                registry of backbones and models (as UniAV)
  utils/                     training helpers
    train_utils.py           seed, optimizer, EMA, one training epoch, validation (R@0.5, CIDEr, Acc, mAP)
    nms.py                   1D soft-NMS (numpy)
    cider.py                 CIDEr metric (same numbers as pycocoevalcap)
    lr_schedulers.py         linear warm-up then cosine decay
    run_record.py            run record: experiments/runs/<host>-<time>.json
tools/                       data preparation, not needed for training once data/ is filled
  htstep_data.py             build: step text vectors; fetch: official files, step vectors and features
  iv2_text.py                InternVideo2 text encoder (text -> 512-d vector), used by htstep_data.py build
  pipeline/                  video download + feature extraction for HT-Step, see "Feature pipeline" below
data/htstep/                 (not in git) official repo clone (data/annotations.json, data/taxonomy.csv, evaluation/),
                             step_emb_iv2j.npz, k400_feats/<video_id>.npz
ckpt/                        (not in git) checkpoints; ckpt/internvideo2/ holds the extractor weights
experiments/runs/            (created by train.py) one JSON record per training run
```

## Setup

    uv venv -p 3.13 .venv
    VIRTUAL_ENV=.venv uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple
    source .venv/bin/activate

On Colab the default runtime already has Python 3.13 and PyTorch; only `pip install huggingface_hub` is needed.

## Data

    python tools/htstep_data.py fetch

clones the official [HT-Step repo](https://github.com/facebookresearch/htstep) into `data/htstep` (annotations of
the train / val_seen splits, taxonomy, evaluation script), downloads `step_emb_iv2j.npz` (InternVideo2 text vectors
of `"<activity>: <headline>"` for every step, made by `python tools/htstep_data.py build`) from
`nguyenminh04/htstep-text-feats`, and the k400_f16 features of the annotated videos from `nguyenminh04/htstep-feats-k400`
(train) and `nguyenminh04/htstep-val-seen-feats` (val_seen) into `data/htstep/k400_feats/`. One file per video:
`k768` (InternVideo2-1B K400 f16, 4 fps, ~4 s window) and `a768` (BEATs audio, zeros if the video has no audio), one
row per second. Videos without features are skipped. Needs `HF_TOKEN` in the environment or `.env`.

## Train

    python train.py                               # one GPU
    torchrun --nproc_per_node=4 train.py          # 4 GPUs: one process each, batch_size videos per GPU
    python train.py --eval ckpt/htstep/model_best.pth.tar   # score a checkpoint

Every epoch prints, on val_seen: the loss, R@0.5 (GT steps matched by one of the top-k predictions with IoU >= 0.5),
CIDEr of the captions (headlines) of the matched steps, Acc (matched steps given the right step) and mAP (official
article grounding mAP, tIoU 0.3-0.7, same numbers as `data/htstep/evaluation/eval_article_grounding.py` on the
videos that have features). The checkpoint with the best mAP goes to `ckpt/htstep/model_best.pth.tar` (R@0.5 is
only logged: HT-Step leaves many real steps unannotated, and R@0.5 counts them as false detections). Training stops
when neither mAP nor CIDEr has improved for `--patience` epochs
(default 3). Other parameters are edited in `libs/core/config.py`.

The omni head (an extra embedding loss against OmniRetriever step vectors, 3584-d) is off while
`dataset.omni_emb_file` is `''`; set it to an npz laid out as `step_emb_iv2j.npz` (`ids`, `emb`) to turn it on.

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
