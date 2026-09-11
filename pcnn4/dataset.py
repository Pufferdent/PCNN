"""JSONL choice dataset: one sample per (situation, solve, play order).

Sample schema (one JSON object per line, written by build_dataset.py):
    sid         situation id, "<replay>_<seq index>"
    pc          PC number 1-7
    stream      the situation's 11-piece stream (pre-held piece first)
    color       40-char broken-board string, row 0 = top
    ordinal     40 ints, 0 = empty, 1..10 = placement ordinal of the cell
    label       1 = the player's chosen solve, 0 = a skipped alternative
    weight      V*-derived sample weight
    save        save piece this solve leaves
    vstar       V* of that save's next segment
    d           vstar - vstar(chosen)
    play_order  piece chars in play order, e.g. "ILJOSZTZLO"
    chosen      bool, solve-level (same for all orders of the solve)
    played      bool, true only for the order the player actually played
"""
import hashlib
import json
import random

import torch
from torch.utils.data import Dataset

from pcnn4.config import DATASET_CONFIG
from pcnn4.encoding import grids_to_tensors, mirror_sample


def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _sid_hash(sid):
    return int(hashlib.md5(sid.encode()).hexdigest()[:8], 16) / 0xffffffff


def split_by_situation(samples, val_frac=None):
    """Deterministic train/val split keeping every sample of a situation on
    the same side (a solve and its alternatives must never straddle the
    split)."""
    if val_frac is None:
        val_frac = DATASET_CONFIG['val_frac']
    train, val = [], []
    for s in samples:
        (val if _sid_hash(s['sid']) < val_frac else train).append(s)
    return train, val


class PCChoiceDataset(Dataset):
    def __init__(self, samples, mirror_prob=0.0, seed=None):
        self.samples = samples
        self.mirror_prob = mirror_prob
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        color, ordinal = s['color'], s['ordinal']
        if self.mirror_prob and self.rng.random() < self.mirror_prob:
            color, ordinal = mirror_sample(color, ordinal)
        color_idx, ordinal_idx = grids_to_tensors(color, ordinal)
        return {
            'color': color_idx,
            'ordinal': ordinal_idx,
            'label': torch.tensor(float(s['label'])),
            'weight': torch.tensor(float(s['weight'])),
        }


def collate(batch):
    return {
        'color': torch.stack([b['color'] for b in batch]),
        'ordinal': torch.stack([b['ordinal'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'weight': torch.stack([b['weight'] for b in batch]),
    }
