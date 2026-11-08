"""Extract InternVideo2 features of the HT-Step chunks in shards.csv: one video row and one audio row per second.

  python tools/pipeline/extract_features.py                                   # every chunk of shards.csv
  python tools/pipeline/extract_features.py --csv tools/pipeline/shards_part1.csv --upload   # this machine's half
  python tools/pipeline/extract_features.py --csv '' --videos path/to/mp4s   # any folder of .mp4, no CSV

Every setting is in CONFIG below; any of them can be overridden on the command line
(--model stage2_f4, --batch-size 4, --audio-ckpt '' ...). Output: <out>/<video_id>.npz, float16, n_sec rows.

The videos of chunk C are <videos>/C/<id>.mp4, as written by download_hf.py (shards already on HF) and
download_videos.py (from YouTube); a chunk is taken once its folder has a .complete file (the downloaders
write it), so the extractor can run beside them with --wait; chunks not downloaded are listed and skipped. With --upload, each
finished chunk goes to HF as feats/htstep_feats_C.tar (its .npz + failed.txt) in the model's
repo (MODELS: nguyenminh04/htstep-feats-k400 for k400_f16).

Two video models (CONFIG['model']):
  k400_f16   InternVideo2-Stage1-1B fine-tuned on K710 then K400, 16 frames per window (4 fps, ~4 s)
             k768 (n_sec, 768)  pooled vision vector, input of the K400 classifier (action-aware)
  stage2_f4  InternVideo2-Stage2-1B, video-text model, 4 frames per window (2 fps, ~2 s)
             v768 (n_sec, 768)  pooled vision vector
             v512 (n_sec, 512)  v768 projected to the text space, L2-normalised
             Same windows and numbers as the older tools/extract_internvideo2.py (the v768 features on HF).
Audio (both models), skipped when audio_ckpt is '':
  a768 (n_sec, 768)  BEATs vector of the 3 s around each second (zeros if the video has no audio)

The training config picks the rows with dataset.iv2_video_keys (['k768'] or ['v768']) and
dataset.iv2_folder. Videos already in --out are skipped, so a stopped run just resumes. With --upload, the .npz of
a chunk's tar already on HF (a partial upload) are copied into --out first, so another machine can resume it.

Needs: ffmpeg on PATH, torch, timm, einops, and the upstream repo cloned unmodified:
  git clone --depth 1 https://github.com/OpenGVLab/InternVideo
Checkpoints (HF, accept the licence on the model page first):
  OpenGVLab/InternVideo2-Stage1-1B-224p-K400  1B_ft_k710_ft_k400_f16.pth
  OpenGVLab/InternVideo2-Stage2_1B-224p-f4    InternVideo2-stage2_1b-224p-f4.pt
  OpenGVLab/InternVideo2-Stage2-6B-Audio      audio_6b.pth (only the BEATs part of the 6B model)
"""
import argparse
import csv
import glob
import io
import os
import shutil
import subprocess
import sys
import tarfile
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

CONFIG = {
    # ---- model ----
    'model': 'k400_f16',                   # 'k400_f16' (16 frames, action-aware) or 'stage2_f4' (4 frames, video-text)
    # ---- what to extract ----
    'csv': os.path.join(HERE, 'shards.csv'),   # chunks to extract (one row per chunk); '' = every .mp4 under videos
    'chunks': [],                          # these chunk numbers only, e.g. ['1000', '1001']; [] = every row
    'upload': False,                       # upload each finished chunk to HF (needs HF_TOKEN)
    'delete_videos': False,                # with upload: delete the chunk's videos once uploaded (small disks)
    'wait': False,                         # keep waiting for chunks still downloading (run it beside the downloaders)
    'upload_every': 0,                     # with upload: also upload the chunk's npz so far (partial tar) every N extracted videos; 0 = off
    'skip_ids': '',                        # text file, one video id per line: already extracted elsewhere, never extract again
    # ---- paths ----
    'videos': 'htstep_videos',             # <videos>/<chunk>/<id>.mp4; file name without .mp4 = video id
    'out': 'iv2_feats',                    # output folder of .npz files
    'repo': 'InternVideo',                 # clone of github.com/OpenGVLab/InternVideo (model code only)
    'ckpt_dir': 'ckpt/internvideo2',       # folder with the checkpoints listed in MODELS
    'audio_ckpt': 'ckpt/internvideo2/audio_6b.pth',   # BEATs weights; '' = no audio
    # ---- sampling: the frames per window and fps come with the model (MODELS) ----
    'size': 224,                           # frames resized (squashed) to size x size, as in training
    'sample_rate': 16000,                  # audio resampled to 16 kHz mono
    # ---- speed ----
    'batch_size': 8,                       # windows per GPU forward; k400_f16 at 8 needs ~3 GB, stage2_f4 takes 32
    'decode_ahead': 4,                     # videos decoded by ffmpeg threads while the GPU works (~200 MB RAM each at 2 fps)
    'limit': 0,                            # only the first N videos (a quick test); 0 = all
}

# frames: frames per window (fixed by the checkpoint); fps: decoding rate, so a window spans frames / fps seconds;
# repo: HF dataset that receives the chunks as feats/htstep_feats_<chunk>.tar (a separate repo per model)
MODELS = {
    'k400_f16': {'ckpt': '1B_ft_k710_ft_k400_f16.pth', 'frames': 16, 'fps': 4, 'repo': 'nguyenminh04/htstep-feats-k400'},
    'stage2_f4': {'ckpt': 'InternVideo2-stage2_1b-224p-f4.pt', 'frames': 4, 'fps': 2, 'repo': 'nguyenminh04/htstep-feats'},
}

MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet normalisation, as in InternVideo2
STD = np.array([0.229, 0.224, 0.225], np.float32)

# BEATs iter3 settings (the cfg of BEATs_iter3_plus_AS2M.pt); audio_6b.pth stores only the weights
BEATS_CFG = {
    'encoder_layers': 12, 'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072,
    'encoder_attention_heads': 12, 'activation_fn': 'gelu', 'dropout': 0.0, 'attention_dropout': 0.0,
    'activation_dropout': 0.0, 'encoder_layerdrop': 0.0, 'dropout_input': 0.0, 'layer_norm_first': False,
    'conv_bias': False, 'conv_pos': 128, 'conv_pos_groups': 16, 'relative_position_embedding': True,
    'num_buckets': 320, 'max_distance': 800, 'gru_rel_pos': True, 'deep_norm': True, 'input_patch_size': 16,
    'layer_wise_gradient_decay_ratio': 0.6, 'embed_dim': 512, 'finetuned_model': False,
    'predictor_dropout': 0.0, 'predictor_class': 527,
}


# ======================================= MODELS =======================================================
class EasyDict(dict):
    __getattr__ = dict.get


def namespace(name, path=None, **attrs):
    """An empty module: lets us import one upstream file without running the package __init__ files
    (they import every model and flash_attn), or stands in for a flash_attn piece we never call."""
    m = types.ModuleType(name)
    if path:
        m.__path__ = [path]
    m.__dict__.update(attrs)
    sys.modules.setdefault(name, m)


def import_upstream(repo):
    iv2 = os.path.join(repo, 'InternVideo2')
    mm = os.path.join(iv2, 'multi_modality', 'models')
    namespace('models', mm)
    namespace('models.backbones', mm + '/backbones')
    namespace('models.backbones.internvideo2', mm + '/backbones/internvideo2')
    namespace('models.backbones.beats', mm + '/backbones/beats')
    namespace('models.backbones.internvideo2.flash_attention_class', FlashAttention=torch.nn.Module)
    # the single-modality (Stage1) code lives in another 'models' package: import it as 'sm_models'
    namespace('sm_models', os.path.join(iv2, 'single_modality', 'models'))
    namespace('sm_models.flash_attention_class', FlashAttention=torch.nn.Module)
    namespace('flash_attn', '')
    namespace('flash_attn.modules', '')
    namespace('flash_attn.modules.mlp', FusedMLP=None)
    namespace('flash_attn.ops', '')
    namespace('flash_attn.ops.rms_norm', DropoutAddRMSNorm=None)


def sdpa_attention(self, x):
    """InternVideo2's Attention._naive_attn with PyTorch's fused kernel: same maths, faster, less memory."""
    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
    if self.qk_normalization:
        q = self.q_norm(q.transpose(1, 2).flatten(-2)).view(B, N, self.num_heads, -1).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2).flatten(-2)).view(B, N, self.num_heads, -1).transpose(1, 2)
    x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(x))


def read_state_dict(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('module', 'model'):   # checkpoints wrap the weights differently
        sd = sd.get(key, sd)
    return sd


def load_k400_f16(ckpt, dev):
    """Stage1 1B fine-tuned on Kinetics: x (B, 3, 16, 224, 224) -> 768-d vector before the classifier."""
    from sm_models import internvideo2 as iv2
    iv2.Attention._naive_attn = sdpa_attention
    enc = iv2.internvideo2_1B_patch14_224(num_frames=16, num_classes=400, use_flash_attn=False,
                                          use_fused_rmsnorm=False, use_fused_mlp=False)
    missing, unexpected = enc.load_state_dict(read_state_dict(ckpt), strict=False)
    print('k400_f16: missing %s, unexpected %s' % (missing, unexpected), flush=True)
    assert not missing, 'checkpoint does not match the model'
    enc = enc.to(dev).half().eval()

    def forward(x):
        tokens = enc.patch_embed(x)                                            # (B, 16, 256, 1408)
        tokens = torch.cat([enc.cls_token.expand(len(x), -1, -1), tokens.flatten(1, 2)], 1) + enc.pos_embed
        for blk in enc.blocks:
            tokens = blk(tokens)
        return {'k768': enc.fc_norm(enc.clip_projector(tokens))}               # what the K400 head reads
    return forward


def load_stage2_f4(ckpt, dev):
    """Stage2 1B (video-text): x (B, 3, 4, 224, 224) -> v768 and its text-space projection v512."""
    from models.backbones.internvideo2 import internvideo2 as iv2
    iv2.Attention._naive_attn = sdpa_attention
    enc = iv2.pretrain_internvideo2_1b_patch14_224(EasyDict(vision_encoder=EasyDict(
        clip_embed_dim=768, use_flash_attn=False, use_fused_rmsnorm=False, use_fused_mlp=False,
        num_frames=4, tubelet_size=1, sep_image_video_pos_embed=True, use_checkpoint=False,
        checkpoint_num=0, clip_teacher_embed_dim=3200, clip_teacher_final_dim=768,
        clip_norm_type='l2', clip_return_layer=6, clip_student_return_interval=1, pretrained=None)))
    proj = torch.nn.Linear(768, 512)
    sd = read_state_dict(ckpt)
    part = lambda p: {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    enc.load_state_dict(part('vision_encoder.'), strict=False)   # the missing keys are training-only decoders
    proj.load_state_dict(part('vision_proj.'))
    enc, proj = enc.to(dev).half().eval(), proj.to(dev).half().eval()

    def forward(x):
        v768 = enc.clip_projector(enc(x, None, False, x_vis_only=True)).reshape(len(x), -1)
        return {'v768': v768, 'v512': F.normalize(proj(v768).float(), dim=-1)}
    return forward


def load_audio_model(cfg, dev):
    from models.backbones.beats.BEATs import BEATs, BEATsConfig
    model = BEATs(BEATsConfig(BEATS_CFG))
    model.load_state_dict(torch.load(cfg['audio_ckpt'], map_location='cpu', weights_only=False), strict=True)
    return model.to(dev).eval()


# ======================================= DECODING =======================================================
def ffmpeg(args):
    return subprocess.run(['ffmpeg', '-v', 'error'] + args + ['-'], capture_output=True).stdout


def decode(path, fps, cfg):
    """frames (n, size, size, 3) uint8 at fps, wav (samples,) float32 mono."""
    s = cfg['size']
    frames = np.frombuffer(ffmpeg(['-i', path, '-vf', 'fps=%d,scale=%d:%d' % (fps, s, s),
                                   '-f', 'rawvideo', '-pix_fmt', 'rgb24']), np.uint8).reshape(-1, s, s, 3)
    wav = np.frombuffer(ffmpeg(['-i', path, '-vn', '-ac', '1', '-ar', str(cfg['sample_rate']), '-f', 'f32le']),
                        np.float32)
    return frames, wav


# ======================================= FEATURES =======================================================
@torch.no_grad()
def video_features(forward, frames, spec, cfg, dev):
    """Second i -> `frames` frames at `fps` around it -> {key: (n_sec, dim)}.
    stage2_f4: times i-0.5 .. i+1 (2 s); k400_f16: times i-1.75 .. i+2 (4 s)."""
    n, fps = spec['frames'], spec['fps']
    n_sec = max(1, int(np.ceil(len(frames) / fps)))
    idx = np.arange(n_sec)[:, None] * fps + np.arange(n) - (n // 2 - 1)    # (n_sec, n) frame indices
    idx = idx.clip(0, len(frames) - 1)                                      # repeat edge frames
    out = {}
    for s in range(0, n_sec, cfg['batch_size']):
        x = (frames[idx[s:s + cfg['batch_size']]].astype(np.float32) / 255 - MEAN) / STD   # (B, n, H, W, 3)
        x = torch.from_numpy(x).permute(0, 4, 1, 2, 3).to(dev).half()                      # (B, 3, n, H, W)
        for k, v in forward(x).items():
            out.setdefault(k, []).append(v.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


@torch.no_grad()
def audio_features(beats, wav, n_sec, cfg, dev):
    """Second i -> the 3 s of audio [i-1, i+2) -> BEATs tokens averaged -> a768."""
    sr = cfg['sample_rate']
    wav = np.pad(wav, (sr, max(0, n_sec * sr - len(wav)) + 3 * sr))         # zeros before 0 s and after the end
    win = np.stack([wav[i * sr:(i + 3) * sr] for i in range(n_sec)])         # (n_sec, 3 * sr)
    out = []
    for s in range(0, n_sec, cfg['batch_size']):
        x = torch.from_numpy(win[s:s + cfg['batch_size']]).to(dev)
        tokens, _ = beats.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))
        out.append(tokens.mean(dim=1).float().cpu())
    return torch.cat(out).numpy()


# ======================================= MAIN =======================================================
def chunk_list(cfg):
    """(ready, waiting): ready = [(chunk, {video id: path})] in CSV order, waiting = chunks whose download is not
    finished (no .complete file in the folder). Without a CSV: one unnamed chunk with every .mp4 under videos."""
    if not cfg['csv']:
        paths = glob.glob(os.path.join(cfg['videos'], '**', '*.mp4'), recursive=True)
        return [('', {os.path.basename(p)[:-4]: p for p in paths})], []
    with open(cfg['csv']) as f:
        rows = list(csv.DictReader(f))
    if cfg['chunks']:
        rows = [r for r in rows if r['chunk'] in cfg['chunks']]
    ready, waiting = [], []
    for r in rows:
        folder = os.path.join(cfg['videos'], r['chunk'])
        if os.path.exists(os.path.join(folder, '.complete')):
            ready.append((r['chunk'], {f[:-4]: os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp4')}))
        else:
            waiting.append(r['chunk'])
    return ready, waiting


def upload_chunk(chunk, ids, failed, cfg, partial=False):
    """<feats>/htstep_feats_<chunk>.tar on HF: the chunk's .npz files + failed.txt.
    partial: chunk not finished; add partial.txt (ids still missing), no .uploaded marker. The final upload overwrites it."""
    from huggingface_hub import HfApi
    tar = os.path.join(cfg['out'], '.htstep_feats_%s.tar' % chunk)
    with tarfile.open(tar, 'w') as t:
        for v in sorted(ids):
            if os.path.exists(os.path.join(cfg['out'], v + '.npz')):
                t.add(os.path.join(cfg['out'], v + '.npz'), arcname=v + '.npz')
        info = tarfile.TarInfo('failed.txt')
        data = '\n'.join(failed).encode()
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
        if partial:
            miss = '\n'.join(sorted(v for v in ids if not os.path.exists(os.path.join(cfg['out'], v + '.npz'))
                                    and v not in cfg['skip'] and not any(v in f for f in failed))).encode()
            info = tarfile.TarInfo('partial.txt')
            info.size = len(miss)
            t.addfile(info, io.BytesIO(miss))
    path = 'feats/htstep_feats_%s.tar' % chunk
    HfApi(token=hf_token()).upload_file(path_or_fileobj=tar, path_in_repo=path, repo_id=MODELS[cfg['model']]['repo'],
                                        repo_type='dataset', commit_message='%s (extract_features.py)' % path)
    os.remove(tar)
    if not partial:
        open(os.path.join(cfg['out'], '.uploaded', chunk), 'w').close()   # a rerun does not upload it again
    print('UPLOADED %s%s' % (path, ' (partial)' if partial else ''), flush=True)


def fetch_uploaded(chunk, cfg):
    """With upload: copy the .npz of the chunk's tar already on HF (a partial upload, maybe from another machine)
    into out, so they are not extracted again and the final upload of this machine still contains them."""
    from huggingface_hub import hf_hub_download
    try:
        tar = hf_hub_download(MODELS[cfg['model']]['repo'], 'feats/htstep_feats_%s.tar' % chunk, repo_type='dataset',
                              token=hf_token(), local_dir=os.path.join(cfg['out'], '.hf'))
    except Exception:   # nothing on HF yet
        return
    n = 0
    with tarfile.open(tar) as t:
        for m in t.getmembers():
            if m.name.endswith('.npz') and not os.path.exists(os.path.join(cfg['out'], m.name)):
                t.extract(m, cfg['out'])
                n += 1
    os.remove(tar)
    print('chunk %s: %d npz taken from the tar on HF' % (chunk, n), flush=True)


def hf_token():
    tok = os.environ.get('HF_TOKEN')
    env = os.path.join(ROOT, '.env')
    if not tok and os.path.exists(env):
        for line in open(env):
            if line.startswith('HF_TOKEN='):
                tok = line.strip().split('=', 1)[1]
    assert tok, 'HF_TOKEN not set'
    return tok


def main(cfg):
    dev = torch.device('cuda')
    spec = MODELS[cfg['model']]
    os.makedirs(os.path.join(cfg['out'], '.uploaded'), exist_ok=True)
    skip = cfg['skip'] = set(open(cfg['skip_ids']).read().split()) if cfg['skip_ids'] else set()
    done = lambda v: v in skip or os.path.exists(os.path.join(cfg['out'], v + '.npz'))   # skipped ids are extracted elsewhere
    uploaded = lambda c: os.path.exists(os.path.join(cfg['out'], '.uploaded', c))
    import_upstream(cfg['repo'])
    load = load_k400_f16 if cfg['model'] == 'k400_f16' else load_stage2_f4
    forward = load(os.path.join(cfg['ckpt_dir'], spec['ckpt']), dev)
    beats = load_audio_model(cfg, dev) if cfg['audio_ckpt'] else None

    t0, secs, count, finished = time.time(), 0, 0, set()
    with ThreadPoolExecutor(cfg['decode_ahead']) as pool:
        while True:
            ready, waiting = chunk_list(cfg)
            # a chunk is finished once extracted (and uploaded, with --upload); a rerun skips it
            ready = [(c, p) for c, p in ready if c not in finished and not (cfg['upload'] and c and uploaded(c))]
            if not ready:
                if waiting and cfg['wait'] and not cfg['limit']:
                    print('%s waiting for %d chunks to finish downloading: %s' % (
                        time.strftime('%H:%M:%S'), len(waiting), ' '.join(waiting[:6])), flush=True)
                    time.sleep(60)
                    continue
                if waiting:
                    print('%d chunks not downloaded (download_hf.py / download_videos.py first): %s' % (
                        len(waiting), ' '.join(waiting)), flush=True)
                break
            n_todo = count + sum(not done(v) for _, paths in ready for v in paths)
            if cfg['limit']:
                n_todo = min(n_todo, cfg['limit'])
            print('%d chunks ready, %d videos to extract with %s -> %s' % (
                len(ready), n_todo - count, cfg['model'], cfg['out']), flush=True)
            for chunk, paths in ready:
                if cfg['upload'] and chunk:
                    fetch_uploaded(chunk, cfg)
                todo = sorted(v for v in paths if not done(v))[:max(0, n_todo - count)]
                failed = {}   # video id -> FAIL / SKIP line, for failed.txt
                # keep `decode_ahead` videos decoding in the background while the GPU extracts the current one
                submit = lambda i: pool.submit(decode, paths[todo[i]], spec['fps'], cfg)
                jobs = [submit(i) for i in range(min(cfg['decode_ahead'], len(todo)))]
                for i, vid in enumerate(todo):
                    frames, wav = jobs[i].result()
                    jobs[i] = None   # free the decoded frames
                    if i + cfg['decode_ahead'] < len(todo):
                        jobs.append(submit(i + cfg['decode_ahead']))
                    count += 1
                    if len(frames) == 0:
                        failed[vid] = 'SKIP %s: no frames decoded' % vid
                        print(failed[vid], flush=True)
                        continue
                    try:   # one bad video must not stop the run
                        feats = video_features(forward, frames, spec, cfg, dev)
                        n_sec = len(next(iter(feats.values())))
                        feats['a768'] = (audio_features(beats, wav, n_sec, cfg, dev) if beats is not None and len(wav)
                                         else np.zeros((n_sec, 768), np.float32))
                    except Exception as e:
                        failed[vid] = 'FAIL %s: %r' % (vid, e)
                        print(failed[vid], flush=True)
                        continue
                    np.savez(os.path.join(cfg['out'], vid + '.npz'), **{k: v.astype(np.float16) for k, v in feats.items()})
                    secs += n_sec
                    if cfg['upload'] and cfg['upload_every'] and count % cfg['upload_every'] == 0:
                        upload_chunk(chunk, paths, failed.values(), cfg, partial=True)
                    el = time.time() - t0
                    print('%d/%d %s/%s: %d s | %.1f video-s per s | ETA %.0f min' % (
                        count, n_todo, chunk, vid, n_sec, secs / el, (n_todo - count) * el / count / 60), flush=True)
                if cfg['upload'] and all(done(v) or v in failed for v in paths):   # the whole chunk went through
                    upload_chunk(chunk, paths, failed.values(), cfg)
                    if cfg['delete_videos']:   # small disks (Kaggle, Colab): the videos can be fetched again
                        shutil.rmtree(os.path.join(cfg['videos'], chunk))
                finished.add(chunk)
                if cfg['limit'] and count >= n_todo:
                    break
            if cfg['limit']:
                break
    print('done', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in CONFIG.items():   # every CONFIG entry becomes an option: 'batch_size' -> --batch-size
        opt = '--' + k.replace('_', '-')
        if isinstance(v, bool):
            p.add_argument(opt, action='store_true', default=v)
        elif isinstance(v, list):
            p.add_argument(opt, nargs='*', default=v)
        else:
            p.add_argument(opt, type=type(v), default=v)
    main(vars(p.parse_args()))
