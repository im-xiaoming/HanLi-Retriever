# HT-Step article grounding (UniAV + OmniRetriever distillation)

Given an instructional video and its article (the steps of a task), the model finds when each step happens. It also
predicts the article (617 articles) and an OmniRetriever-like vector (3584-d) for every segment.

Backbone: [UniAV](https://arxiv.org/abs/2404.03179) (video: InternVideo2 K400 features, audio: BEATs, cross-attention,
6-level pyramid). Heads: article conditioning, event + boundary, step choice, article head, omni distillation (KL +
cosine to the OmniRetriever `encode_av` vector of each segment).

## Results (val_seen)

| mAP (article given) | ArtAcc | OmniCos | OmniCosC | OmniR@1 |
|---|---|---|---|---|
| 30.37 | 59.8 | 91.1 | 75.2 | 60.8 |

Details: [docs/htstep_report.md](docs/htstep_report.md), every run: [docs/experiments_report.md](docs/experiments_report.md).

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
       python train.py --eval ckpt/htstep/model_best.pth.tar

   On Colab: `bash tools/colab_train.sh` (setup + data + train).

All parameters are in `libs/core/config.py`. Training is deterministic (same code, data and seed give the same
numbers); about 12 epochs, ~45 min on an A100.

## Code

```
train.py              training / evaluation
libs/core/config.py   all parameters
libs/datasets/        HT-Step dataset, file readers, crop, training targets
libs/modeling/        backbone, heads, losses, model (event_archs.py)
libs/utils/           optimizer, EMA, train / valid loop, NMS, metrics
tools/                data preparation (features, text vectors); pipeline/ = video download + feature extraction
```
