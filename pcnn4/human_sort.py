#!/usr/bin/env python3
"""Drag-to-sort + gap-size segment weighting (successor to the pairwise
page, which was too much clicking).

One draggable ranked list per PC number — 463 items total (568 segments
minus PC2's 105 dupes, which get manually assigned values). Order stays
editable the whole time; each row (except the top one) carries a gap chip
saying how big the step down from the row above is:

    =  x1      >  x1.2     >>  x2     >>>  x5     >>>>  x20

Values chain multiplicatively from 100 at the top and are shown live.

Default order = PCReview v3's leftover-quality judgement
(``pcreview.pcname.PcName``: top/mid/bad, defined for PCs 2-6); ties —
and PCs 1/7, which PCReview does not classify — fall back to robot V*
descending.

Usage:
    /usr/bin/python3 -m pcnn4.human_sort --make-html
    /usr/bin/python3 -m pcnn4.human_sort --compute human_sort_results.json
"""
import argparse
import json
import os
import sys

from pcnn4.config import PATHS
from pcnn4 import v3bridge  # noqa: F401
from pcnn4.human_pref import (is_dupe, load_groups, load_names,
                              write_weights)

PCREVIEW_DIR = os.environ.get(
    'PCREVIEW_V3_PATH',
    os.path.normpath(os.path.join(PATHS['root'], '..', 'PCReview v3')))
if PCREVIEW_DIR not in sys.path:
    sys.path.append(PCREVIEW_DIR)

GAP_STEPS = [1.0, 1.2, 2.0, 5.0, 20.0]
TIER_RANK = {'top': 0, 'mid': 1, 'bad': 2}

HTML_OUT = os.path.join(PATHS['data_dir'], 'human_sort.html')
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'templates', 'human_sort.html')


def pcreview_tier(key, pc):
    """PCReview v3's top/mid/bad judgement of a segment; None for PC 1/7."""
    from pcreview.pcname import PcName
    return PcName(key, pc).tier()


def gen_lists():
    """{pc: [{'k', 'name', 'tier', 'vstar'}]} in default order:
    PCReview tier first, then V* descending. PC2 dupes are excluded
    (manually valued)."""
    from pc_value import load_table
    names = load_names()
    groups = {}
    for key, row in load_table().items():
        pc = (row['round'] - 1) % 7 + 1
        if pc == 2 and is_dupe(key):
            continue
        tier = pcreview_tier(key, pc)
        groups.setdefault(pc, []).append(
            {'k': key, 'name': names[key], 'tier': tier,
             'vstar': row['vstar']})
    for pc, items in groups.items():
        items.sort(key=lambda it: (TIER_RANK.get(it['tier'], 1),
                                   -it['vstar']))
    return {pc: groups[pc] for pc in sorted(groups)}


def make_html(out=HTML_OUT):
    lists = gen_lists()
    with open(TEMPLATE) as f:
        template = f.read()
    page = template.replace(
        '__DATA__', json.dumps(lists, separators=(',', ':')))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write(page)
    return out, lists


def compute_from_state(state, lists=None):
    """Page export -> per-pc weight tables (chained gap products,
    top = 100). ``state``: {pc: {'order': [keys], 'gaps': [floats]}}."""
    lists = lists or gen_lists()
    results = {}
    for pc, items in lists.items():
        expected = {it['k'] for it in items}
        st = state.get(str(pc)) or state.get(pc)
        results[pc] = {'answered': 0, 'total': len(items),
                       'rms_log10': None, 'weights': {}}
        if not st:
            results[pc]['weights'] = {k: None for k in expected}
            continue
        order, gaps = st['order'], st['gaps']
        if set(order) != expected or len(gaps) != len(order) - 1:
            raise ValueError(
                'pc%s export does not match the current segment list '
                '(regenerated page with different defaults?)' % pc)
        v = 100.0
        weights = {}
        for i, key in enumerate(order):
            if i > 0:
                v /= gaps[i - 1]
            weights[key] = v
        results[pc]['weights'] = weights
        results[pc]['answered'] = len(order)
    # unranked segments (PC2 dupes) still get report rows, weight None,
    # awaiting their manual values
    for pc, keys in load_groups().items():
        if pc in results:
            for k in keys:
                results[pc]['weights'].setdefault(k, None)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--make-html', action='store_true')
    ap.add_argument('--compute', metavar='RESULTS_JSON')
    args = ap.parse_args()

    if args.make_html:
        out, lists = make_html()
        counts = {pc: len(v) for pc, v in lists.items()}
        print('items per pc: %s (total %d)' %
              (counts, sum(counts.values())))
        print('page: %s' % out)
    elif args.compute:
        with open(args.compute) as f:
            data = json.load(f)
        state = data.get('state', data)
        results = compute_from_state(state)
        xlsx, js = write_weights(results)
        for pc, res in sorted(results.items()):
            print('pc%d: %d/%d ranked' %
                  (pc, res['answered'], res['total']))
        print('wrote %s and %s' % (xlsx, js))
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
