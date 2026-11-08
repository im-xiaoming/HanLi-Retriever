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
        'num_classes': 48,       # at most 48 GT segments per video (only 1 of the 18072 videos has more)
        'center_sample_radius': 0.0,   # > 0: event positives only within this many strides of a step centre
        'omni_emb_dim': 3584,
        'train_fraction': 1.0,   # < 1: train on a fixed random subset of the train videos (learning curves)
    },

    'model': {
        # condition the pyramid features on the article's step texts (ActionFormer-T of the HT-Step paper)
        'article_cond': True,      # pc_v6: mAP 30.50 with, ~25.5 without (article given)
        # predict the video's article (617 recipes) from the video, so it needs not be given at inference
        'article_head': True,
        # article_head only: 617 = every article gets its own learned vector (zero at start) on top of its text;
        # 0 = the article is its text only
        'article_learned': 617,    # +5-7 ArtAcc over the text only
        # omni: True = a learned head compresses a (start, end) segment into its omni vector (attention pooling of
        # the level-0 features + MLP), trained on the GT segments; OmniCos is then measured with it
        'omni_span_head': True,
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
        'omni_head_dim': 3584,   # size of the OmniRetriever vectors; 0 = no omni head
        'regression_range': [[0, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 10000]],
        'head_num_layers': 3,
        'head_kernel_size': 3,
        'head_with_ln': True,
        'use_abs_pe': True,
        'max_seq_len': 256,
        'nmax': 48,
        'clip_dim': 512,
        'train_cfg': {
            'loss_weight_reg': 1.0,
            'loss_weight_iou': 1.0,
            'loss_weight_emb': 0.2,
            'loss_weight_omni_emb': 0.2,
            'omni_bank_size': 4096,       # omni head: candidate negatives kept (mined: the weakest are dropped)
            'omni_temperature': 0.05,     # omni head: softmax temperature of the distillation loss
            # omni head: weight of 1 - cos(student, teacher). InfoNCE alone ignores the part every teacher shares
            # (the train mean already has cos 0.80 with a val teacher), so the student stays far from its teacher
            'loss_weight_omni_cos': 2.0,   # 2.0 keeps OmniCos ~86 (1.0 drifts down to 80)
            'loss_weight_omni_span_nce': 0.2,   # omni_span_head only: InfoNCE of the segment vectors
            'loss_weight_omni_span_cos': 2.0,   # omni_span_head only: 1 - cos of the segment vectors
            'loss_weight_span': 0.2,
            'loss_weight_article': 0.5,   # article_head only
            'article_detach': False,      # article_head only: True = no article gradient into the backbone
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
            # pick the step of a segment among the steps of the video's article (HT-Step article grounding);
            # False = among all 4958 steps
            'article': True,
            # article_head only: use the predicted article instead of the given one (conditioning and step choice)
            'pred_article': False,
            'pred_article_topk': 1,   # pred_article only: the candidates are the steps of the top-k articles
            # article_head only: True = every step competes, score + article_soft_weight * log p(its article | video)
            # (no hard choice of one article; overrides 'article' and 'pred_article' for the step choice)
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
