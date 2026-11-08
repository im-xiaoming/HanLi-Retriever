#!/bin/bash
# PC side: keep asking for a free Colab A100; once granted, upload the files and start tools/pipeline/colab_dl_extract.sh there.
# Usage: set -a; . ./.env; set +a; [CSV=tools/pipeline/rest2_colab.csv] tools/pipeline/colab_dl_launch.sh   (HOME=~/colabN picks the Google account; the default HOME has the A100)
cd "$(dirname "$0")/../.."
C=/home/minh/.local/bin/colab
S=${SESSION:-a100}   # an existing session of this name is reused
CSV=${CSV:-tools/pipeline/rest_colab.csv}   # this machine's share
$C sessions 2>&1 | grep -qw "$S" || $C new --gpu ${GPU:-A100} -s $S >/dev/null 2>&1 || { echo "no A100 for $HOME"; exit 1; }
echo "$(date +%T) $S granted"
printf %s "$HF_TOKEN" > /tmp/hf_token_colab
$C upload -s $S /tmp/hf_token_colab /content/hf_token && rm /tmp/hf_token_colab
$C upload -s $S tools/pipeline/colab_dl_extract.sh /content/colab_dl_extract.sh
$C upload -s $S tools/pipeline/run.sh /content/run.sh
$C upload -s $S tools/pipeline/extract_features.py /content/extract_features.py
$C upload -s $S tools/pipeline/download_videos.py /content/download_videos.py
$C upload -s $S tools/pipeline/download_hf.py /content/download_hf.py
$C upload -s $S $CSV /content/$(basename $CSV)
echo "import subprocess; subprocess.Popen('CSV=tools/pipeline/$(basename $CSV) nohup bash /content/colab_dl_extract.sh > /content/boot.log 2>&1 &', shell=True)" | $C exec -s $S
setsid nohup bash tools/pipeline/colab_keepalive.sh $S >/dev/null 2>&1 &   # via bash: the file may lack the exec bit
sleep 20; pgrep -f "[c]olab_keepalive.sh $S" >/dev/null && echo "keepalive running" || echo "WARNING: keepalive is not running, the session will be shut down when idle"
echo "$(date +%T) started"
