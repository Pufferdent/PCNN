#!/usr/bin/env python3
"""Build the PC-NN-v4 choice dataset from v3's parsed replay corpus.

For every full 4L PC in the human replays:
  1. reconstruct the 11-piece stream (v3 build_weights machinery),
  2. enumerate every solve of that stream with sfinder (cached per stream),
  3. annotate each solve's save with its next-PC V*,
  4. keep solves inside the V* window around the player's chosen solve
     (d = V*_solve - V*_chosen > -v_window),
  5. expand each kept solve into ALL of its playable piece orders,
  6. emit one JSONL sample per (solve, order): broken-board color grid +
     placement-ordinal grid + label (chosen or not) + V*-derived weight.

The player's chosen solve is identified among sfinder's solutions by exact
placement-set match in original (pre-clear) coordinates; a situation whose
chosen solve cannot be matched is skipped and counted.

Usage:
    /usr/bin/python3 -m pcnn4.build_dataset [--limit-files N]
        [--limit-seqs N] [--max-solutions N] [--max-orders N] [--out PATH]
"""
import argparse
import csv
import json
import os
import sys

# sfinder's path CSV packs every solve fumen of a pattern into one
# ';'-joined field, which blows past the default 128 KiB csv field limit.
csv.field_size_limit(sys.maxsize)

from pcnn4.config import PATHS, BUILD_CONFIG, WEIGHT_CONFIG, PIECE_TYPES
from pcnn4 import v3bridge  # noqa: F401  (puts v3 pc-nn on sys.path)
from pcnn4.encoding import (EncodingError, grids_from_order,
                            replay_original_placements)
from pcnn4.weighting import include, negative_weight, positive_weight_each

from build_weights import (EMPTY_BOARD_FUMEN, infer_round,  # noqa: E402
                           reconstruct_stream)
from pc_value import (end_of_pc_save, load_table, lookup,  # noqa: E402
                      next_segment_key)
from sfinder_parser import (fumen_placed_pieces,  # noqa: E402
                            parse_sfinder_path_csv, run_sfinder)
from solution_order import (GlueDesyncError, enumerate_playable_orders,
                            glue)  # noqa: E402

STAT_KEYS = [
    'files', 'sequences', 'situations_emitted', 'skip_not_full_pc',
    'skip_stream', 'skip_replay_desync', 'skip_sfinder_failed',
    'skip_chosen_not_found',
    'skip_chosen_no_order', 'skip_no_negatives', 'sol_no_vstar',
    'sol_windowed_out', 'sol_glue_failed', 'sol_no_playable_order',
    'sol_capped', 'orders_capped', 'samples_pos', 'samples_neg',
]


def solutions_for_stream(stream, timeout, use_cache=True):
    """All sfinder path solve fumens for ``stream`` on the empty board,
    cached per stream (identical streams recur across replays)."""
    cache_dir = PATHS['sfinder_cache_dir']
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, stream + '.json')
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    csv_path = run_sfinder(
        EMPTY_BOARD_FUMEN, stream, clear_lines=4, hold=True,
        output_base=os.path.join(PATHS['sfinder_output_dir'], stream),
        timeout=timeout)
    fumens = [s['fumen'] for s in parse_sfinder_path_csv(csv_path)]
    with open(cache_path, 'w') as f:
        json.dump(fumens, f)
    return fumens


def unique_orders(initial, placements, stream, max_orders):
    """Playable orders deduped by placement sequence (hold-path variants of
    the same physical order collapse). Returns (orders as k-tuples, capped)."""
    orders = enumerate_playable_orders(
        initial, placements, stream, hold_enabled=True, max_orders=max_orders)
    seen = {}
    for o in orders:
        ks = tuple(r['k'] for r in o)
        seen.setdefault(ks, True)
    capped = max_orders is not None and len(orders) >= max_orders
    return list(seen), capped


def process_sequence(sid, seq, pc, stream, table, args, stats):
    """Returns a list of sample dicts for one PC situation (may be empty)."""
    try:
        chosen_placements = replay_original_placements(seq['steps'])
    except (EncodingError, KeyError) as e:
        stats['skip_replay_desync'] += 1
        print("  %s: replay desync (%s)" % (sid, e), file=sys.stderr)
        return []
    chosen_set = frozenset(chosen_placements)
    chosen_multiset = tuple(sorted(p for p, _ in chosen_placements))

    try:
        fumens = solutions_for_stream(stream, args.sfinder_timeout,
                                      use_cache=not args.no_cache)
    except Exception as e:  # sfinder timeout/failure: skip the situation
        stats['skip_sfinder_failed'] += 1
        print("  %s: sfinder failed for %s (%s)" %
              (sid, stream, type(e).__name__), file=sys.stderr)
        return []

    # Annotate every solution with its save and V*; find the chosen one.
    entries = []
    chosen_entry = None
    for fumen in fumens:
        placed = fumen_placed_pieces(fumen)
        save = end_of_pc_save(stream, placed)
        row = lookup(next_segment_key(pc, stream, save), table)
        if row is None:
            stats['sol_no_vstar'] += 1
            continue
        entry = {'fumen': fumen, 'save': save, 'vstar': row['vstar'],
                 'glued': None}
        entries.append(entry)
        if chosen_entry is None and \
                tuple(sorted(placed)) == chosen_multiset:
            try:
                entry['glued'] = glue(fumen)
            except GlueDesyncError:
                stats['sol_glue_failed'] += 1
                continue
            if frozenset(entry['glued'][1]) == chosen_set:
                chosen_entry = entry

    if chosen_entry is None:
        stats['skip_chosen_not_found'] += 1
        print("  %s: chosen solve not among sfinder solutions" % sid,
              file=sys.stderr)
        return []
    v_chosen = chosen_entry['vstar']

    # V* window filter, then cap negatives only (never the chosen solve).
    neg_entries = []
    for entry in entries:
        if entry is chosen_entry:
            continue
        if include(entry['vstar'] - v_chosen):
            neg_entries.append(entry)
        else:
            stats['sol_windowed_out'] += 1
    if args.max_solutions and len(neg_entries) > args.max_solutions:
        stats['sol_capped'] += len(neg_entries) - args.max_solutions
        print("  %s: capping %d in-window alternatives to %d" %
              (sid, len(neg_entries), args.max_solutions), file=sys.stderr)
        # keep the most informative negatives: highest V* (largest weight)
        neg_entries.sort(key=lambda e: e['vstar'], reverse=True)
        neg_entries = neg_entries[:args.max_solutions]

    pos_samples, neg_samples = [], []
    for entry in [chosen_entry] + neg_entries:
        is_chosen = entry is chosen_entry
        d = entry['vstar'] - v_chosen
        if entry['glued'] is None:
            try:
                entry['glued'] = glue(entry['fumen'])
            except GlueDesyncError:
                stats['sol_glue_failed'] += 1
                continue
        initial, placements = entry['glued']
        order_ks_list, capped = unique_orders(
            initial, placements, stream, args.max_orders)
        if capped:
            stats['orders_capped'] += 1
        if not order_ks_list:
            stats['sol_no_playable_order'] += 1
            if is_chosen:
                stats['skip_chosen_no_order'] += 1
                print("  %s: chosen solve has no playable order" % sid,
                      file=sys.stderr)
                return []
            continue

        w_solve = 1.0 if is_chosen else negative_weight(d)
        if WEIGHT_CONFIG['order_weight_mode'] == 'split':
            w_order = w_solve / len(order_ks_list)
        else:
            w_order = w_solve
        for ks in order_ks_list:
            ordered = [placements[k] for k in ks]
            try:
                color, ordinal = grids_from_order(placements, ks)
            except EncodingError:
                stats['sol_no_playable_order'] += 1
                break
            sample = {
                'sid': sid, 'pc': pc, 'stream': stream,
                'color': color, 'ordinal': ordinal,
                'label': 1 if is_chosen else 0,
                'weight': w_order,
                'save': entry['save'], 'vstar': entry['vstar'], 'd': d,
                'play_order': ''.join(p for p, _ in ordered),
                'chosen': is_chosen,
                'played': is_chosen and ordered == chosen_placements,
            }
            (pos_samples if is_chosen else neg_samples).append(sample)

    if not pos_samples:
        return []
    if not neg_samples:
        stats['skip_no_negatives'] += 1

    # Balance: the chosen solve's orders share pos_ratio * total negative
    # weight (the lone positive must not drown among its negatives).
    neg_total = sum(s['weight'] for s in neg_samples)
    w_pos = positive_weight_each(len(pos_samples), neg_total)
    for s in pos_samples:
        s['weight'] = w_pos

    stats['samples_pos'] += len(pos_samples)
    stats['samples_neg'] += len(neg_samples)
    stats['situations_emitted'] += 1
    return pos_samples + neg_samples


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data-dir', default=PATHS['v3_parsed_weighted'])
    ap.add_argument('--out', default=PATHS['dataset'])
    ap.add_argument('--limit-files', type=int, default=None)
    ap.add_argument('--limit-seqs', type=int, default=None,
                    help='Stop after emitting N situations in total')
    ap.add_argument('--max-solutions', type=int,
                    default=BUILD_CONFIG['max_solutions'])
    ap.add_argument('--max-orders', type=int,
                    default=BUILD_CONFIG['max_orders'])
    ap.add_argument('--sfinder-timeout', type=int,
                    default=BUILD_CONFIG['sfinder_timeout'])
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    table = load_table()
    stats = {k: 0 for k in STAT_KEYS}

    files = sorted(f for f in os.listdir(args.data_dir)
                   if f.endswith('.json'))
    if args.limit_files:
        files = files[:args.limit_files]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = False
    with open(args.out, 'w') as out:
        for fi, fn in enumerate(files):
            if done:
                break
            stats['files'] += 1
            with open(os.path.join(args.data_dir, fn)) as f:
                data = json.load(f)
            replay_id = os.path.splitext(fn)[0].replace('_parsed', '')
            cum_pieces = 0
            for si, seq in enumerate(data.get('sequences', [])):
                stats['sequences'] += 1
                sid = '%s_%d' % (replay_id, si)
                placed = [st.get('placed_piece') for st in seq['steps']]
                cum_pc = (cum_pieces * 5) % 7 + 1
                cum_pieces += len(seq['steps'])
                if len(placed) != 10 or \
                        any(p not in PIECE_TYPES for p in placed):
                    stats['skip_not_full_pc'] += 1
                    continue
                variants = reconstruct_stream(seq)
                if not variants:
                    stats['skip_stream'] += 1
                    continue
                pc, stream = infer_round(variants, cum_pc,
                                         seq.get('pc_number', 1))
                if pc is None:
                    stats['skip_stream'] += 1
                    continue
                samples = process_sequence(sid, seq, pc, stream, table,
                                           args, stats)
                for s in samples:
                    out.write(json.dumps(s) + '\n')
                if samples:
                    print("[%d/%d] %s: %d samples (%d pos)" %
                          (fi + 1, len(files), sid, len(samples),
                           sum(1 for s in samples if s['label'])),
                          file=sys.stderr)
                if args.limit_seqs and \
                        stats['situations_emitted'] >= args.limit_seqs:
                    done = True
                    break

    meta = {'stats': stats, 'weight_config': WEIGHT_CONFIG,
            'args': {k: v for k, v in vars(args).items()}}
    with open(PATHS['dataset_meta'], 'w') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(stats, indent=1), file=sys.stderr)
    print("Wrote %s" % args.out, file=sys.stderr)


if __name__ == '__main__':
    main()
