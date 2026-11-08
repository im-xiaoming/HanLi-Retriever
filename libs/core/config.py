# Every parameter lives here. Edit this file to change one; there is no config file and only
# --eval / --patience on the command line.

cfg = {
    'output_folder': './ckpt/htstep',
    'init_rand_seed': 1234567891,
    'num_workers': 2,
    'batch_size': 32,            # per GPU: with torchrun on N GPUs one step sees N * batch_size videos

    'dataset': {
        # official HT-Step files (git clone https://github.com/facebookresearch/htstep data/htstep)
        'json_file': './datasets/annotations.json',
        'taxonomy_file': './datasets/taxonomy.csv',
        # InternVideo2 text vectors (512-d) of the 4958 steps, "<activity>: <headline>" (tools/htstep_data.py steps)
        'step_emb_file': './datasets/step_emb_iv2j.npz',
        # OmniRetriever encode_av vector (3584-d) of every GT segment: train/<id>.npz and val/<id>.npz
        # (tools/fetch_htstep.py; HF htstep-omni-av = train, htstep-val-seen-feats omni_av/ = val_seen)
        'omni_emb_folder': './datasets/omni_av_feats',
        # <video_id>.npz: InternVideo2 K400 f16 video k768 + BEATs audio a768, one row per second
        'feat_folder': './datasets/k400_feats',
        'default_fps': 16,
        'max_seq_len': 256,      # every video is resampled to 256 steps
        'trunc_thresh': 0.5,
        'crop_ratio': [0.9, 1.0],
        'num_classes': 48,       # at most 48 GT segments per video (none of the videos with features has more)
        'omni_emb_dim': 3584,
    },

    'model': {
        # article head: every article (recipe) gets a learned vector (zero at start) on top of its text vector
        'num_articles': 617,
        'pool_queries': 8,          # omni span head: learned queries of its attention pooling
        'omni_span_hidden': 4096,   # hidden size of the omni span head MLP
        'backbone_arch': [2, 3, 5],
        'scale_factor': 2,
        'input_dim_V': 768,
        'input_dim_A': 768,
        'n_head': 4,
        'embd_kernel_size': 3,
        'embd_dim': 512,
        'embd_with_ln': True,
        'head_dim': 512,
        'omni_head_dim': 3584,   # size of the OmniRetriever vectors
        'regression_range': [[0, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 10000]],
        'head_num_layers': 3,
        'head_kernel_size': 3,
        'head_with_ln': True,
        'use_abs_pe': True,
        'max_seq_len': 256,
        'clip_dim': 512,
        'train_cfg': {
            'loss_weight_reg': 1.0,
            'loss_weight_iou': 1.0,
            'loss_weight_emb': 0.2,
            # omni (point head and span head, the same weights for both): KL distillation of the teacher's
            # similarity distribution over the teachers of the batch
            'loss_weight_omni_kd': 0.2,
            # 1 - cos(student, teacher). The KL alone ignores the part every teacher shares (the train mean already
            # has cos 0.80 with a val teacher), so the student would stay far from its teacher
            'loss_weight_omni_cos': 2.0,
            'omni_temperature': 0.05,      # softmax temperature of the omni KL distillation (0.02 / 0.1: worse)
            'loss_weight_span': 0.2,
            'loss_weight_article': 0.5,
            'cls_prior_prob': 0.01,
            'dropout': 0.0,
            'droppath': 0.1,
        },
        'test_cfg': {
            'score_thresh': 0.05,
            'duration_thresh': 0.05,
            'max_seg_num': 200,     # 200 with nms_sigma 0.75: mAP 25.70 vs 25.27 at (100, 0.5)
            'iou_power': 0.3,
            'nms_sigma': 0.75,      # 0.75: +0.25 mAP over 0.5 on val_seen (best-mAP checkpoint, 2026-11-01)
            # the step of a segment is picked among the steps of the video's article (given: HT-Step article
            # grounding). True = the article is not given: every step competes,
            # score + article_soft_weight * log p(its article | video) from the article head
            'article_soft': False,
            'article_soft_weight': 0.5,   # 0.5: mAP 18.14 vs 17.63 at 1.0 and 16.09 for the hard top-1 (pc_v3)
        },
    },

    'train_cfg': {
        'clip_grad_l2norm': 1.0,
        'ema_decay': 0.999,
    },

    'opt': {
        'learning_rate': 0.0003,   # batch 32: 1e-4 at batch 4 scaled by sqrt(8)
        'clip_proj_lr': 0.003,
        'weight_decay': 0.0001,
        'epochs': 10,
        'warmup_epochs': 2,
    },
}
