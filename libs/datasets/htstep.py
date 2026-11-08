import os
import csv
import json
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset, make_generator
from .data_utils import truncate_feats, label_points

from .helper import load_step_vectors, load_omni_segments, load_normalize, filtering
import logging

logger = logging.getLogger(__name__)


@register_dataset("htstep")
class HTStepDataset(Dataset):
    def __init__(
        self,
        is_training,      # True: training split (random crops)
        split,            # list of subsets to keep, e.g. ['train']
        feat_folder,      # feature folder, one <id>.npz per video
        json_file,        # official HT-Step annotations: {id: {subset, duration, activity, annotations: [{segment, id}]}}
        taxonomy_file,    # taxonomy.csv: one row per step (global_step_index, activity, variation, headline)
        step_emb_file,    # text vectors of every step, "<activity>: <headline>"
        omni_emb_folder,  # OmniRetriever segment vectors: <folder>/train/<id>.npz and <folder>/val/<id>.npz
        default_fps,      # nominal fps (16), converts seconds <-> steps
        max_seq_len,      # steps per video (256)
        trunc_thresh,     # a cropped segment is kept only if at least this fraction is left (0.5)
        crop_ratio,       # (0.9, 1.0): a random crop keeps 90-100% of the video
        num_classes: int,      # at most this many GT segments per video (48)
        scale_factor,     # length ratio between two pyramid levels
        regression_range, # regression range of every pyramid level
        backbone_arch,    # (#conv, #stem, #pyramid branch), gives the number of levels
        center_sample_radius=0.0,  # > 0: a point is positive only within this many strides of the segment centre
        train_fraction=1.0,        # training split only: keep this fraction of the videos (learning curves)
        omni_emb_dim=3584
    ):
        self.feat_folder = feat_folder
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.num_classes = num_classes
        self.center_sample_radius = center_sample_radius
        self.omni_emb_dim = omni_emb_dim
        # the train and val_seen omni vectors live in separate folders, so val never leaks into training
        omni_part = 'train' if 'train' in split else 'val'
        self.omni_folder = os.path.join(omni_emb_folder, omni_part)

        
        # This line converts each row in CSV file into dictionary.
        with open(taxonomy_file, newline='') as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r['global_step_index']))
            
        self.pool_ids = [int(r['global_step_index']) for r in rows]
        self.step_row = {s: i for i, s in enumerate(self.pool_ids)}
        self.pool_text = [r['headline'].strip() for r in rows]
        
        self.pool_emb = load_step_vectors(step_emb_file, self.step_row)   # (4958, 512)
        
        
        # article = activity + variation (one recipe); its steps are the candidates of a video at inference
        self.article_steps, self.step_article = {}, {}
        for i, r in enumerate(rows):
            self.article_steps.setdefault((r['activity'], r['variation']), []).append(i)
            self.step_article[i] = (r['activity'], r['variation'])
        self.articles = list(self.article_steps)   # 617 articles; a video's 'article' is its index here
        self.article_idx = {a: i for i, a in enumerate(self.articles)}

        # read the annotations, keep the videos of the split
        self.data_list = self._load_json_db(json_file)
        if is_training and train_fraction < 1.0:
            # a fixed random subset (same seed every run), to measure how the scores grow with the data
            rng = np.random.RandomState(0)
            keep = sorted(rng.choice(len(self.data_list), int(round(train_fraction * len(self.data_list))), replace=False))
            self.data_list = tuple(self.data_list[i] for i in keep)

        # point generator of the pyramid
        self.fpn_strides = [scale_factor**i for i in range(backbone_arch[-1] + 1)] # [1, 2, 4, 8, 16, 32]
        self.point_generator = make_generator(
            'point',
            **{
                'regression_range': regression_range,
                'max_seq_len_ori': max_seq_len,
                'max_buffer_len_factor': 1.0,
                'fpn_levels': len(self.fpn_strides),
                'scale_factor': scale_factor,
                'max_div_factor': self.fpn_strides[-1],
            }
        )

    def _load_json_db(self, json_file):
        with open(json_file, 'r') as f:
            json_db = json.load(f)
            
        dict_db = tuple()
        for key, value in json_db.items():
            if value['subset'].lower() not in self.split:
                continue
            
            # keep the segments with a positive length, at most num_classes of them;
            # ann_idx remembers where each kept segment sits in the json list (the omni files use that index)
            ann_idx = [i for i, a in enumerate(value['annotations']) if a['segment'][1] > a['segment'][0]]
            ann_idx = ann_idx[:self.num_classes]
            anns = [value['annotations'][i] for i in ann_idx]
            if len(anns) == 0:
                continue
            
            path = os.path.join(self.feat_folder, key + '.npz')
            if not os.path.exists(path):
                continue
            
            try:
                with np.load(path) as z:
                    rows = min(z['k768'].shape[0], z['a768'].shape[0])
            except Exception as e:
                logger.exception(f'Excepion error: {e}')
                continue
            
            # one feature row per second: many more rows than the annotated duration means the downloaded video
            # is another version of it (re-upload, intro added), so the annotated times do not match it
            duration = value.get('duration')
            if duration and rows > duration + max(5.0, 0.05 * duration):
                continue

            segments = np.array([a['segment'] for a in anns], dtype=np.float32) # (N, 2)
            if segments[:, 0].min() >= rows:
                continue   # features cut short: every GT step starts after they end
            
            steps = np.array([self.step_row[a['id']] for a in anns], dtype=np.int64)
            dict_db += ({'id': key,
                         'fps': self.default_fps,
                         'duration': value['duration'],
                         'segments': segments, # (n, 2)
                         'labels': np.arange(len(anns), dtype=np.int64),   # segment index; (n,)
                         # {index in the json 'annotations' list: slot (= label) of that segment}
                         'ann_slots': {json_index: slot for slot, json_index in enumerate(ann_idx)},
                         'steps': steps,   # step (pool row) of every segment; (n,)
                         'activity': value['activity'],
                         # every step of the video's article (all its segments share one article)
                         'cand': np.array(self.article_steps[self.step_article[int(steps[0])]], dtype=np.int64),
                         'article': self.article_idx[self.step_article[int(steps[0])]],
            }, )
        return dict_db

    def omni_teacher_mean(self):
        """Mean of the OmniRetriever vectors of every GT segment of this split, (3584,) float32: the part every
        segment shares. The OmniCosC metric subtracts it (the train mean alone already has cos ~0.80 with a val
        teacher). Zeros if the split has no omni vectors."""
        total = np.zeros((self.omni_emb_dim,), dtype=np.float64)
        count = 0
        for video_item in self.data_list:
            omni_vectors, omni_ok = load_omni_segments(
                os.path.join(self.omni_folder, video_item['id'] + '.npz'),
                video_item['ann_slots'],
                video_item['steps'],
                self.step_row,
                self.num_classes,
                self.omni_emb_dim,
            )
            total += omni_vectors[omni_ok].sum(axis=0)
            count += int(omni_ok.sum())
        return (total / max(count, 1)).astype(np.float32)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        vid_id = video_item['id']

        # read the features, L2-normalise every row (a768 is all zeros for videos without audio)
        fv, fa = load_normalize(self.feat_folder, vid_id + '.npz')
        t = min(fv.shape[0], fa.shape[0])

        feat_stride = float(t * self.default_fps) / self.max_seq_len
        num_frames = feat_stride
        feat_offset = 0.5
        fv = fv[:t].transpose(0, 1).contiguous() # (768, T)
        fa = fa[:t].transpose(0, 1).contiguous() # (768, T)
        #                   (1, 768, T)
        fv = F.interpolate(fv[None], size=self.max_seq_len, mode='linear', align_corners=False)[0] # (768, 256)
        fa = F.interpolate(fa[None], size=self.max_seq_len, mode='linear', align_corners=False)[0] # (768, 256)

        segments = torch.from_numpy(video_item['segments'] * video_item['fps'] / feat_stride - feat_offset) # (n, 2)
        labels = torch.from_numpy(video_item['labels']) # (n,)
        
        if self.is_training:
            segments, labels = filtering(segments, labels, self.max_seq_len, feat_offset, self.trunc_thresh)

        data_dict = {'video_id': vid_id,
                     'feats': {'visual': fv, 'audio': fa},
                     'segments': segments,
                     'labels': labels,
                     'fps': video_item['fps'],
                     'duration': video_item['duration'],
                     'feat_stride': feat_stride,
                     'feat_num_frames': num_frames}

        # random crop of the features during training
        if self.is_training:
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio)

        # training targets of every pyramid point
        points = self.point_generator(self.fpn_strides, data_dict['feats']['visual'], self.is_training)

        data_dict['gt_cls_labels'], data_dict['gt_offsets'] = label_points(
            points, data_dict['segments'], data_dict['labels'], self.num_classes, self.center_sample_radius)
        data_dict['points'] = points

        # step of every GT segment (pool row), padded to num_classes with -1
        sidx = np.full((self.num_classes,), -1, dtype=np.int64)
        sidx[:len(video_item['steps'])] = video_item['steps']
        
        data_dict['step_idx'] = torch.from_numpy(sidx) # (48,)

        # teacher of the omni head: the OmniRetriever vector of every GT segment, in slot order (slot = label).
        # Read after the random crop: the crop drops segments but never renumbers the slots.
        omni_vectors, omni_ok = load_omni_segments(
            os.path.join(self.omni_folder, vid_id + '.npz'),
            video_item['ann_slots'],
            video_item['steps'],
            self.step_row,
            self.num_classes,
            self.omni_emb_dim,
        )
        data_dict['omni_seg'] = torch.from_numpy(omni_vectors)   # (48, 3584)
        data_dict['omni_ok'] = torch.from_numpy(omni_ok)         # (48,) True = the slot has a teacher vector
        # a number unique to every GT segment of the split, so the candidate bank can skip duplicates
        data_dict['seg_key'] = torch.arange(self.num_classes) + idx * self.num_classes   # (48,)

        data_dict['cand'] = torch.from_numpy(video_item['cand'])   # steps of the video's article, ~8
        data_dict['article'] = video_item['article']   # index in self.articles, target of the article head
        
        data_dict['activity'] = video_item['activity']
        data_dict['gt_steps'] = video_item['steps'].tolist()
        data_dict['cap_text'] = [self.pool_text[s] for s in video_item['steps']]   # GT headlines
        
        # original boundaries in seconds, only used for scoring
        data_dict['segments_sec'] = video_item['segments'].tolist()
        return data_dict
