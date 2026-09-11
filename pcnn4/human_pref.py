#!/usr/bin/env python3
"""Human preference weights for next-PC segments (replacing robot V*).

AHP-style pairwise-comparison workflow over the same 568 segment scenarios
the V* table scores, grouped by PC number ((round-1) % 7 + 1; R1/R8 both
feed PC 1):

1. ``--make-html`` emits an interactive comparison page
   (``data/human_pref.html``) and an Excel scaffold
   (``data/human_pref_scaffold.xlsx``). Coverage per PC number: every
   non-dupe pair, plus each dupe (segment with a doubled piece) vs 5
   seeded-random non-dupes.
2. Answer comparisons in the page (autosaves to localStorage; export JSON
   when done) — or fill the scaffold's ``ratio`` column in Excel
   (left:right, e.g. ``5`` or ``1/5``).
3. ``--compute <results.json|filled.xlsx>`` solves log10 least-squares on
   each PC number's comparison graph and writes:
   - ``data/human_rounds.xlsx``: per-PC weights (best = 100) + consistency
   - ``data/human_rounds.json``: key -> weight table, same keying as
     vstar_rounds.json, for wiring into build_dataset later.
"""
import argparse
import json
import os
import random

from pcnn4.config import PATHS
from pcnn4 import v3bridge  # noqa: F401

SEED = 7
RATIO_STEPS = [1, 2, 3, 5, 10, 20, 50, 100]

HTML_OUT = os.path.join(PATHS['data_dir'], 'human_pref.html')
SCAFFOLD_OUT = os.path.join(PATHS['data_dir'], 'human_pref_scaffold.xlsx')
WEIGHTS_XLSX = os.path.join(PATHS['data_dir'], 'human_rounds.xlsx')
WEIGHTS_JSON = os.path.join(PATHS['data_dir'], 'human_rounds.json')
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'templates', 'human_pref.html')


def is_dupe(key):
    return len(set(key)) < len(key)


def load_groups():
    """{pc_number: sorted [segment keys]} from the v3 V* table."""
    from pc_value import load_table
    groups = {}
    for key, row in load_table().items():
        pc = (row['round'] - 1) % 7 + 1
        groups.setdefault(pc, []).append(key)
    return {pc: sorted(ks) for pc, ks in groups.items()}


def load_names():
    """{segment key: round-sheet type name} (e.g. 'full', 'no LS', 'T>S')."""
    from pc_value import load_table
    return {key: row['type'] for key, row in load_table().items()}


DUPE_PARTNERS = 5


def gen_comparisons(seed=SEED):
    """Deterministic comparison list: [{'id', 'pc', 'a', 'b'}].

    Per PC number (independent rng per PC, so tweaking one PC's rule never
    reshuffles the others): all non-dupe pairs first (shuffled — they anchor
    the weight scale), then each dupe vs 5 random partners. Partners are
    non-dupes when enough exist; PC 1 has only one non-dupe (the full bag),
    so its dupes are topped up with random *other dupes* to reach 5.
    PC 2 emits no dupe comparisons at all — its dupe segments get manually
    assigned values instead.
    """
    comparisons = []
    for pc, keys in sorted(load_groups().items()):
        rng = random.Random(seed * 1000 + pc)
        nds = [k for k in keys if not is_dupe(k)]
        dupes = [k for k in keys if is_dupe(k)]
        core = [(a, b) for i, a in enumerate(nds) for b in nds[i + 1:]]
        rng.shuffle(core)
        extra = []
        if pc != 2:
            seen = set()
            for d in dupes:
                partners = rng.sample(nds, min(DUPE_PARTNERS, len(nds)))
                short = DUPE_PARTNERS - len(partners)
                if short > 0:
                    pool = [x for x in dupes if x != d]
                    partners += rng.sample(pool, min(short, len(pool)))
                for p in partners:
                    pair = frozenset((d, p))
                    if pair not in seen:
                        seen.add(pair)
                        extra.append((d, p))
            rng.shuffle(extra)
        for i, (a, b) in enumerate(core + extra):
            comparisons.append(
                {'id': 'pc%d-%03d' % (pc, i), 'pc': pc, 'a': a, 'b': b})
    return comparisons


def make_html(comparisons, out=HTML_OUT):
    with open(TEMPLATE) as f:
        template = f.read()
    page = template.replace('__COMPARISONS__',
                            json.dumps(comparisons, separators=(',', ':')))
    page = page.replace('__NAMES__',
                        json.dumps(load_names(), separators=(',', ':')))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write(page)
    return out


def make_scaffold(comparisons, out=SCAFFOLD_OUT):
    """Excel alternative to the page: one row per comparison, fill 'ratio'
    with left:right preference (e.g. 5, or 1/5 = prefer right 5:1)."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'comparisons'
    ws.append(['id', 'pc', 'left segment', 'right segment',
               'ratio (left:right, e.g. 5 or 1/5)'])
    for c in comparisons:
        ws.append([c['id'], c['pc'], c['a'], c['b'], None])
    wb.save(out)
    return out


def read_results(path):
    """{comparison id: ratio (left/right, float)} from the page's JSON
    export or a filled scaffold xlsx. Blank/1 entries count as 1:1;
    missing rows are simply unanswered."""
    if path.endswith('.xlsx'):
        import openpyxl
        ws = openpyxl.load_workbook(path, read_only=True)['comparisons']
        answers = {}
        for row in list(ws.iter_rows(values_only=True))[1:]:
            cid, raw = row[0], row[4]
            if cid is None or raw is None or str(raw).strip() == '':
                continue
            s = str(raw).strip()
            if '/' in s:
                num, den = s.split('/', 1)
                answers[cid] = float(num) / float(den)
            else:
                answers[cid] = float(s)
        return answers
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and 'answers' in data:
        data = data['answers']
    return {cid: float(r) for cid, r in data.items()}


def compute_weights(comparisons, answers):
    """Per-PC log10 least-squares over the answered comparison graph.

    Returns {pc: {'weights': {key: w}, 'rms_log10': float, 'answered': n,
    'total': n}} with weights scaled so the best segment = 100. Segments
    with no answered comparison get weight None.
    """
    import numpy as np
    import math

    by_pc = {}
    for c in comparisons:
        by_pc.setdefault(c['pc'], []).append(c)

    out = {}
    for pc, comps in sorted(by_pc.items()):
        answered = [c for c in comps if c['id'] in answers]
        keys = sorted({c['a'] for c in answered} | {c['b'] for c in answered})
        idx = {k: i for i, k in enumerate(keys)}
        result = {'answered': len(answered), 'total': len(comps),
                  'rms_log10': None, 'weights': {}}
        out[pc] = result
        if not answered:
            continue
        # rows: w_a - w_b = log10(ratio); anchor: mean(w) = 0
        rows, rhs = [], []
        for c in answered:
            r = max(answers[c['id']], 1e-6)
            row = [0.0] * len(keys)
            row[idx[c['a']]] = 1.0
            row[idx[c['b']]] = -1.0
            rows.append(row)
            rhs.append(math.log10(r))
        rows.append([1.0] * len(keys))
        rhs.append(0.0)
        A = np.array(rows)
        b = np.array(rhs)
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = A[:-1] @ w - b[:-1]
        result['rms_log10'] = float(np.sqrt(np.mean(resid ** 2)))
        scaled = 10.0 ** (w - w.max() + 2.0)      # best segment = 100
        result['weights'] = {k: float(scaled[idx[k]]) for k in keys}
    # every segment of every pc appears in the report; uncompared ones
    # (e.g. PC2 dupes, valued manually) carry None
    for pc, keys in load_groups().items():
        if pc in out:
            for k in keys:
                out[pc]['weights'].setdefault(k, None)
    return out


def write_weights(results, xlsx_out=WEIGHTS_XLSX, json_out=WEIGHTS_JSON):
    import openpyxl
    wb = openpyxl.Workbook()
    summary = wb.active
    summary.title = 'summary'
    summary.append(['pc', 'segments', 'answered', 'total comparisons',
                    'rms_log10 (consistency; <0.15 is good)'])
    flat = {}
    for pc, res in sorted(results.items()):
        summary.append([pc, len(res['weights']), res['answered'],
                        res['total'], res['rms_log10']])
        ws = wb.create_sheet('PC%d' % pc)
        ws.append(['segment', 'dupe', 'weight (best=100)'])
        ranked = sorted(res['weights'].items(),
                        key=lambda kv: -(kv[1] if kv[1] is not None else -1))
        for key, weight in ranked:
            ws.append([key, is_dupe(key), weight])
            if weight is not None:
                flat[key] = {'pc': pc, 'weight': weight}
    wb.save(xlsx_out)
    with open(json_out, 'w') as f:
        json.dump(flat, f, indent=1)
    return xlsx_out, json_out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--make-html', action='store_true')
    ap.add_argument('--compute', metavar='RESULTS',
                    help='results JSON from the page, or filled scaffold xlsx')
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    comparisons = gen_comparisons(args.seed)
    if args.make_html:
        html = make_html(comparisons)
        scaffold = make_scaffold(comparisons)
        print('%d comparisons' % len(comparisons))
        print('page:     %s' % html)
        print('scaffold: %s' % scaffold)
    elif args.compute:
        answers = read_results(args.compute)
        results = compute_weights(comparisons, answers)
        xlsx, js = write_weights(results)
        for pc, res in sorted(results.items()):
            print('pc%d: %d/%d answered, rms_log10=%s' %
                  (pc, res['answered'], res['total'],
                   ('%.3f' % res['rms_log10'])
                   if res['rms_log10'] is not None else 'n/a'))
        print('wrote %s and %s' % (xlsx, js))
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
