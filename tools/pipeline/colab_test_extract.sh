#!/bin/bash
# Colab side: k400_f16 + BEATs features of the HT-Step test videos -> nguyenminh04/htstep-test-feats (feats/<split>.tar),
# uploaded every 10 min; a rerun resumes from the tars on HF. Expects /content/hf_token, extract_features.py and
# test_feats_sync.py in /content. Logs: /content/logs/. Keep the session alive from the PC (colab_keepalive.sh).
cd /content
pkill -f "[t]est_feats_sync.py"; pkill -f "[e]xtract_features.py"; sleep 2
mkdir -p logs
export HF_TOKEN=$(cat /content/hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1
pip install -q huggingface_hub timm einops >/dev/null 2>&1
[ -d InternVideo ] || git clone -q --depth 1 https://github.com/OpenGVLab/InternVideo
python -u test_feats_sync.py setup test_seen > logs/setup_test_seen.log 2>&1 || { echo "setup failed"; exit 1; }
python -u test_feats_sync.py setup test_unseen > logs/setup_test_unseen.log 2>&1 &   # downloads while test_seen extracts
DL=$!
nohup python -u test_feats_sync.py loop 10 > logs/upload.log 2>&1 &
EX="--csv '' --batch-size ${BS:-32} --decode-ahead 8"
eval python -u extract_features.py $EX --videos videos/test_seen --out feats/test_seen > logs/extract_test_seen.log 2>&1
wait $DL
eval python -u extract_features.py $EX --videos videos/test_unseen --out feats/test_unseen > logs/extract_test_unseen.log 2>&1
python -u test_feats_sync.py upload >> logs/upload.log 2>&1
echo "all done $(date +%T)"
