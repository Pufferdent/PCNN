import json

import torch

from pcnn4.dataset import (PCChoiceDataset, collate, load_jsonl,
                           split_by_situation)


def _sample(sid, label, weight, color='N' * 40, ordinal=None):
    return {'sid': sid, 'pc': 1, 'stream': 'TIOLJSZTILO',
            'color': color, 'ordinal': ordinal or [0] * 40,
            'label': label, 'weight': weight, 'save': 'T',
            'vstar': 4352.5, 'd': 0.0, 'play_order': '',
            'chosen': bool(label), 'played': False}


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / 'ds.jsonl'
    samples = [_sample('a_0', 1, 2.0), _sample('a_0', 0, 1.0)]
    with open(path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    loaded = load_jsonl(str(path))
    assert loaded == samples


def test_dataset_and_collate():
    color = 'T' * 10 + 'N' * 30
    ordinal = [1] * 10 + [0] * 30
    ds = PCChoiceDataset([_sample('a_0', 1, 2.5, color, ordinal)])
    item = ds[0]
    assert item['color'].shape == (4, 10)
    assert item['ordinal'].shape == (4, 10)
    batch = collate([item, item])
    assert batch['color'].shape == (2, 4, 10)
    assert batch['label'].tolist() == [1.0, 1.0]
    assert batch['weight'].tolist() == [2.5, 2.5]
    assert batch['color'].dtype == torch.long


def test_mirror_prob_applies():
    color = 'L' + 'N' * 39
    ordinal = [1] + [0] * 39
    ds = PCChoiceDataset([_sample('a_0', 1, 1.0, color, ordinal)],
                         mirror_prob=1.0, seed=0)
    item = ds[0]
    # 'L' at top-left became 'J' at top-right (color index 4 = 'J')
    assert int(item['color'][0, 9]) == 4
    assert int(item['ordinal'][0, 9]) == 1
    assert int(item['color'][0, 0]) == 0


def test_split_keeps_situations_together():
    samples = []
    for i in range(50):
        sid = 'replay%d_0' % i
        samples.append(_sample(sid, 1, 1.0))
        samples.append(_sample(sid, 0, 1.0))
    train, val = split_by_situation(samples, val_frac=0.3)
    assert len(train) + len(val) == len(samples)
    train_sids = {s['sid'] for s in train}
    val_sids = {s['sid'] for s in val}
    assert not (train_sids & val_sids)
    assert val  # 50 sids at 30% should land some in val
    # deterministic
    train2, val2 = split_by_situation(samples, val_frac=0.3)
    assert train2 == train and val2 == val
