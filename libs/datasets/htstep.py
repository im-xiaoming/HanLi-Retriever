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

from typing import Literal


def load_step_vectors(
    path,      # npz with 'ids' (global_step_index) and 'emb' (N, D), tools/htstep_data.py build
    step_row,  # {global_step_index: pool row}
):
    """(4958, D) vectors in pool order."""
    z = np.load(path)
    emb = np.zeros((len(step_row), z['emb'].shape[1]), dtype=np.float32)
    emb[[step_row[int(i)] for i in z['ids']]] = z['emb']
    return emb


@register_dataset("htstep")
class HTStepDataset(Dataset):
    def __init__(
        self,
        is_training,      # True: training split (random crops)
        split: Literal['train', 'val_seen'],            # list of subsets to keep, e.g. ['train']
        feat_folder,      # feature folder, one <id>.npz per video
        video_key,        # key of the video rows in the npz (k768)
        json_file,        # official HT-Step annotations: {id: {subset, duration, activity, annotations: [{segment, id}]}}
        taxonomy_file,    # taxonomy.csv: one row per step (global_step_index, activity, variation, headline)
        step_emb_file,    # text vectors of every step, "<activity>: <headline>"
        omni_emb_file,    # OmniRetriever vectors of every step, same layout; '' = none
        default_fps,      # nominal fps (16), converts seconds <-> steps
        max_seq_len,      # steps per video (256)
        trunc_thresh,     # a cropped segment is kept only if at least this fraction is left (0.5)
        crop_ratio,       # (0.9, 1.0): a random crop keeps 90-100% of the video
        num_classes: int,      # at most this many GT segments per video (48)
        scale_factor,     # length ratio between two pyramid levels
        regression_range, # regression range of every pyramid level
        backbone_arch,    # (#conv, #stem, #pyramid branch), gives the number of levels
        center_sample_radius=0.0,  # > 0: a point is positive only within this many strides of the segment centre
    ):
        self.feat_folder = feat_folder
        self.video_key = video_key
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.num_classes = num_classes
        self.center_sample_radius = center_sample_radius

        # The classes are the 4958 steps of the taxonomy, each defined by its text (open vocabulary, as OV-AVEL).
        # Steps never seen in training are still in the pool. The step ids (global_step_index) run from 0 to 5784
        # with gaps, so the model works with pool rows and pool_ids maps a row back to its step id.
        
        # This line converts each row in CSV file into dictionary.
        with open(taxonomy_file, newline='') as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r['global_step_index']))
            
        self.pool_ids = [int(r['global_step_index']) for r in rows]
        self.step_row = {s: i for i, s in enumerate(self.pool_ids)}
        self.pool_text = [r['headline'].strip() for r in rows]
        self.pool_emb = load_step_vectors(step_emb_file, self.step_row)   # (4958, 512)
        self.pool_omni_emb = load_step_vectors(omni_emb_file, self.step_row) if omni_emb_file else None
        # article = activity + variation (one recipe); its steps are the candidates of a video at inference
        self.article_steps, self.step_article = {}, {}
        for i, r in enumerate(rows):
            self.article_steps.setdefault((r['activity'], r['variation']), []).append(i)
            self.step_article[i] = (r['activity'], r['variation'])
        self.articles = list(self.article_steps)   # 617 articles; a video's 'article' is its index here
        self.article_idx = {a: i for i, a in enumerate(self.articles)}

        # read the annotations, keep the videos of the split
        self.data_list = self._load_json_db(json_file)

        # point generator of the pyramid
        self.fpn_strides = [scale_factor**i for i in range(backbone_arch[-1]+1)] # [1, 2, 4, 8, 16, 32]
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

    def _load_json_db(self, json_file) -> tuple[dict[Literal["id", 'fps', 'duration', 'segments', 'labels', 'steps', 'activity', 'cand']]]:
        with open(json_file, 'r') as f:
            json_db = json.load(f)
        dict_db = tuple()
        for key, value in json_db.items():
            if value['subset'].lower() not in self.split: # split = ['train'] or ['val_seen']
                continue
            anns = [a for a in value['annotations'] if a['segment'][1] > a['segment'][0]][:self.num_classes]
            if len(anns) == 0:
                continue
            path = os.path.join(self.feat_folder, key + '.npz')
            if not os.path.exists(path):
                continue   # not extracted (yet)
            try:
                with np.load(path) as z:
                    rows = min(z[self.video_key].shape[0], z['a768'].shape[0])   # one row per second; reads (CRC-checks) both
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                continue   # broken file
            segments = np.array([a['segment'] for a in anns], dtype=np.float32) # (N, 2)
            if segments[:, 0].min() >= rows:
                continue   # features cut short: every GT step starts after they end
            steps = np.array([self.step_row[a['id']] for a in anns], dtype=np.int64)
            dict_db += ({'id': key,
                         'fps': self.default_fps,
                         'duration': value['duration'], # None
                         'segments': segments, # (n, 2)
                         'labels': np.arange(len(anns), dtype=np.int64),   # segment index; (n,)
                         'steps': steps,   # step (pool row) of every segment; (n,)
                         'activity': value['activity'],
                         # every step of the video's article (all its segments share one article)
                         'cand': np.array(self.article_steps[self.step_article[int(steps[0])]], dtype=np.int64),
                         'article': self.article_idx[self.step_article[int(steps[0])]],
            }, )
        return dict_db

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        vid_id = video_item['id']

        # read the features, L2-normalise every row (a768 is all zeros for videos without audio)
        z = np.load(os.path.join(self.feat_folder, vid_id + '.npz'))
        fv = z[self.video_key].astype(np.float32) # TxC
        fa = z['a768'].astype(np.float32) # TxC
        fv = fv / np.maximum(np.linalg.norm(fv, axis=1, keepdims=True), 1e-6)
        fa = fa / np.maximum(np.linalg.norm(fa, axis=1, keepdims=True), 1e-6)
        t = min(fv.shape[0], fa.shape[0]) # T

        # Time grid. n feature rows (one per second = default_fps frames) are resampled to max_seq_len steps,
        # so one step is feat_stride frames long and step i is centred on frame (i + 0.5) * feat_stride.
        # Time t (seconds) sits at grid position t * fps / feat_stride - 0.5;
        # the model maps back with (position * feat_stride + 0.5 * num_frames) / fps.
        feat_stride = float(t * self.default_fps) / self.max_seq_len # total frame: T * 16
        num_frames = feat_stride
        feat_offset = 0.5
        fv = torch.from_numpy(np.ascontiguousarray(fv[:t].transpose())) # (768, T)
        fa = torch.from_numpy(np.ascontiguousarray(fa[:t].transpose())) # (768, T)
        #                   (1, 768, T)
        fv = F.interpolate(fv[None], size=self.max_seq_len, mode='linear', align_corners=False)[0] # (768, 256)
        fa = F.interpolate(fa[None], size=self.max_seq_len, mode='linear', align_corners=False)[0] # (768, 256)

        segments = torch.from_numpy(video_item['segments'] * video_item['fps'] / feat_stride - feat_offset) # (n, 2)
        labels = torch.from_numpy(video_item['labels']) # (n,)
        if self.is_training:
            # drop GT steps that start after the features end; a step cut at the end is kept only if
            # at least trunc_thresh of it is left; always keep at least one step
            vid_len = self.max_seq_len + feat_offset
            keep = [i for i, (s, e) in enumerate(segments.tolist()) if s < vid_len and e > s and (min(e, vid_len) - s) / (e - s) >= self.trunc_thresh]
            keep = keep or [0]
            segments, labels = segments[keep].clamp(max=vid_len), labels[keep]

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
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio)

        # training targets of every pyramid point
        points = self.point_generator(self.fpn_strides, data_dict['feats']['visual'], self.is_training)

        data_dict['gt_cls_labels'], data_dict['gt_offsets'] = label_points(
            points, data_dict['segments'], data_dict['labels'], self.num_classes, self.center_sample_radius)
        data_dict['points'] = points

        # step of every GT segment (pool row), padded to num_classes with -1
        sidx = np.full((self.num_classes,), -1, dtype=np.int64)
        sidx[:len(video_item['steps'])] = video_item['steps']
        data_dict['step_idx'] = torch.from_numpy(sidx) # (48,)
        data_dict['cand'] = torch.from_numpy(video_item['cand'])   # steps of the video's article, ~8
        data_dict['article'] = video_item['article']   # index in self.articles, target of the article head
        data_dict['activity'] = video_item['activity']
        data_dict['gt_steps'] = video_item['steps'].tolist()
        data_dict['cap_text'] = [self.pool_text[s] for s in video_item['steps']]   # GT headlines
        # original boundaries in seconds, only used for scoring
        data_dict['segments_sec'] = video_item['segments'].tolist()
        return data_dict
