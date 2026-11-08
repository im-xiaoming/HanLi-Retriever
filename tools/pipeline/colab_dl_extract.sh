#!/bin/bash
# Colab side: this machine's share of the HT-Step pipeline (download + k400_f16 extraction + upload to HF), from its CSV.
# Expects /content/hf_token and the pipeline files flat in /content (uploaded by colab_dl_launch.sh). CSV=... picks another share.
cd /content
pkill -f "[t]ools/pipeline/"; sleep 2   # a rerun replaces the old pipeline processes (finished work on disk is kept)
mkdir -p tools/pipeline && mv -f run.sh extract_features.py download_videos.py download_hf.py *.csv tools/pipeline/
export HF_TOKEN=$(cat /content/hf_token)
CSV=${CSV:-tools/pipeline/rest_colab.csv}
pip install -q yt-dlp huggingface_hub >/dev/null 2>&1
command -v deno >/dev/null || curl -fsSL https://deno.land/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.deno/bin:$PATH
[ -d InternVideo ] || git clone -q --depth 1 https://github.com/OpenGVLab/InternVideo
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
for repo, f in (('OpenGVLab/InternVideo2-Stage1-1B-224p-K400', '1B_ft_k710_ft_k400_f16.pth'),
                ('OpenGVLab/InternVideo2-Stage2-6B-Audio', 'audio_6b.pth')):
    print(hf_hub_download(repo, f, local_dir='/content/ckpt/internvideo2', token=os.environ['HF_TOKEN']), flush=True)
# ids already extracted elsewhere (htstep_feats_iv2_train.tar): never extracted again
print(hf_hub_download('nguyenminh04/htstep-feats-k400', 'extracted_videos.txt', repo_type='dataset', local_dir='/content',
                      token=os.environ['HF_TOKEN']), flush=True)
PY
EX_ARGS="${EX_ARGS:---batch-size 32 --decode-ahead 8} --skip-ids /content/extracted_videos.txt" NO_YT=1 SMALL_DISK=1 VIDEOS=/content/videos OUT=/content/feats LOGS=/content/logs PY=python bash tools/pipeline/run.sh $CSV
