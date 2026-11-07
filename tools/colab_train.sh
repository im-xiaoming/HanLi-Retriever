#!/bin/bash
# HT-Step training on a Colab GPU. Run from /content, in the background (nohup), one step at a time:
#
#   BRANCH=htstep-omni-distill bash colab_train.sh setup        # code + data + validation, ~10 min
#   bash colab_train.sh train <run name> [train.py options]      # e.g. --set model.article_head=True
#
# setup
#   1. clones (or updates) the code into /content/UniAV, on $BRANCH
#   2. official HT-Step annotations + taxonomy (github.com/facebookresearch/htstep) -> datasets/
#   3. step text vectors (HF htstep-text-feats) -> datasets/step_emb_iv2j.npz
#   4. k400 features and OmniRetriever segment vectors (tools/fetch_htstep.py) -> datasets/
#   5. tools/validate_htstep.py: stops here (no setup.done) if a blocking problem is found
# The HF token, if any, is read from /content/hf_token (the repos used here are public).
set -e
STEP=$1
BRANCH=${BRANCH:-htstep-omni-distill}
CODE=/content/UniAV
REPO=https://github.com/im-xiaoming/Athena-Retriever.git
if [ -f /content/hf_token ]; then export HF_TOKEN=$(cat /content/hf_token); fi
export HF_HUB_DISABLE_PROGRESS_BARS=1

if [ "$STEP" = "setup" ]; then
  rm -f /content/setup.done
  pip -q install huggingface_hub

  # 1. code
  if [ -d "$CODE/.git" ]; then
    git -C "$CODE" fetch -q origin "$BRANCH"
    git -C "$CODE" checkout -q -B "$BRANCH" "origin/$BRANCH"
  else
    git clone -q -b "$BRANCH" "$REPO" "$CODE"
  fi
  cd "$CODE"
  mkdir -p datasets logs

  # 2. annotations and taxonomy
  if [ ! -f datasets/annotations.json ]; then
    rm -rf /content/htstep_official
    git clone -q --depth 1 https://github.com/facebookresearch/htstep /content/htstep_official
    cp /content/htstep_official/data/annotations.json /content/htstep_official/data/taxonomy.csv datasets/
  fi

  # 3. step text vectors
  python - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download('nguyenminh04/htstep-text-feats', 'text_feats/step_emb_iv2j.npz', repo_type='dataset',
                    token=os.environ.get('HF_TOKEN'))
shutil.copy(p, 'datasets/step_emb_iv2j.npz')
print('step vectors ok')
EOF

  # 4. features
  python tools/fetch_htstep.py --cache /content/hf_cache

  # 5. validation
  python tools/validate_htstep.py --workers 8 --report datasets/validation_report.json
  touch /content/setup.done
  echo "SETUP DONE"

elif [ "$STEP" = "train" ]; then
  RUN=$2
  shift 2
  cd "$CODE"
  mkdir -p /content/logs
  python -u train.py --output "$RUN" "$@" --set num_workers=8 2>&1 | tee "/content/logs/$RUN.log"
  echo "TRAIN EXIT ${PIPESTATUS[0]}" | tee -a "/content/logs/$RUN.log"

else
  echo "usage: colab_train.sh setup | train <run name> [train.py options]"
  exit 1
fi
