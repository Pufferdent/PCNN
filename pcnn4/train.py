#!/usr/bin/env python3
"""Train PCChoiceNet on the v4 choice dataset (weighted BCE).

Not run yet — wired up so training is one command once the dataset is built:
    /usr/bin/python3 -m pcnn4.train --data data/dataset.jsonl
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from pcnn4.config import DATASET_CONFIG, PATHS, TRAIN_CONFIG
from pcnn4.dataset import (PCChoiceDataset, collate, load_jsonl,
                           split_by_situation)
from pcnn4.model import PCChoiceNet


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and \
            torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    tot_loss = tot_w = tot_correct_w = 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            color = batch['color'].to(device)
            ordinal = batch['ordinal'].to(device)
            label = batch['label'].to(device)
            weight = batch['weight'].to(device)

            logits = model(color, ordinal)
            loss = (criterion(logits, label) * weight).sum() / weight.sum()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            w = weight.sum().item()
            tot_loss += loss.item() * w
            tot_w += w
            pred = (logits > 0).float()
            tot_correct_w += (weight * (pred == label).float()).sum().item()
    return tot_loss / max(tot_w, 1e-9), tot_correct_w / max(tot_w, 1e-9)


def train_timed(model, train_loader, val_loader, device, criterion,
                optimizer, args):
    """Wall-clock-bounded training: run for --max-minutes, evaluating on the
    held-out set and checkpointing every --eval-every seconds (checked at
    batch granularity, so intervals and epoch counts are approximate)."""
    max_sec = args.max_minutes * 60.0
    start = last_eval = time.time()
    n_batches = max(1, len(train_loader))
    history = []
    best_val = float('inf')
    step = 0
    run_loss = run_w = 0.0

    def evaluate(tag):
        nonlocal best_val, run_loss, run_w
        va_loss, va_acc = run_epoch(model, val_loader, device, criterion)
        tr_loss = run_loss / max(run_w, 1e-9)
        rec = {'t_sec': round(time.time() - start, 1), 'step': step,
               'epoch_approx': round(step / n_batches, 2),
               'train_loss': tr_loss, 'val_loss': va_loss,
               'val_acc': va_acc}
        history.append(rec)
        print("t=%4.0fs  step %5d (~epoch %5.1f)  train %.4f  "
              "val %.4f/%.3f  %s" %
              (rec['t_sec'], step, rec['epoch_approx'], tr_loss,
               va_loss, va_acc, tag), flush=True)
        ckpt = {'model': model.state_dict(), 'cfg': model.cfg,
                'step': step, 'history': history}
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'v4_last.pt'))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, os.path.join(args.checkpoint_dir, 'v4_best.pt'))
        with open(os.path.join(args.checkpoint_dir, 'history.json'),
                  'w') as f:
            json.dump(history, f, indent=1)
        run_loss = run_w = 0.0

    model.train(True)
    done = False
    while not done:
        for batch in train_loader:
            color = batch['color'].to(device)
            ordinal = batch['ordinal'].to(device)
            label = batch['label'].to(device)
            weight = batch['weight'].to(device)
            logits = model(color, ordinal)
            loss = (criterion(logits, label) * weight).sum() / weight.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            w = weight.sum().item()
            run_loss += loss.item() * w
            run_w += w

            now = time.time()
            if now - start >= max_sec:
                done = True
                break
            if now - last_eval >= args.eval_every:
                evaluate('')
                last_eval = time.time()
                model.train(True)
    evaluate('final')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data', default=PATHS['dataset'])
    ap.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    ap.add_argument('--batch-size', type=int,
                    default=TRAIN_CONFIG['batch_size'])
    ap.add_argument('--lr', type=float, default=TRAIN_CONFIG['lr'])
    ap.add_argument('--weight-decay', type=float,
                    default=TRAIN_CONFIG['weight_decay'])
    ap.add_argument('--val-frac', type=float,
                    default=DATASET_CONFIG['val_frac'])
    ap.add_argument('--mirror-prob', type=float,
                    default=DATASET_CONFIG['mirror_prob'])
    ap.add_argument('--device', default=None)
    ap.add_argument('--checkpoint-dir', default=PATHS['checkpoint_dir'])
    ap.add_argument('--max-minutes', type=float, default=None,
                    help='Wall-clock-bounded run: train this many minutes '
                         'instead of --epochs')
    ap.add_argument('--eval-every', type=float, default=60.0,
                    help='Seconds between eval+checkpoint in timed mode')
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(TRAIN_CONFIG['seed'])

    samples = load_jsonl(args.data)
    train_s, val_s = split_by_situation(samples, args.val_frac)
    print("device=%s  train=%d  val=%d" % (device, len(train_s), len(val_s)))

    train_loader = DataLoader(
        PCChoiceDataset(train_s, mirror_prob=args.mirror_prob,
                        seed=TRAIN_CONFIG['seed']),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(
        PCChoiceDataset(val_s), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate)

    model = PCChoiceNet().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.max_minutes:
        train_timed(model, train_loader, val_loader, device, criterion,
                    optimizer, args)
        return

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    history = []
    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion,
                                    optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': tr_loss,
                        'train_acc': tr_acc, 'val_loss': va_loss,
                        'val_acc': va_acc})
        print("epoch %3d  train %.4f/%.3f  val %.4f/%.3f  (%.1fs)" %
              (epoch, tr_loss, tr_acc, va_loss, va_acc, time.time() - t0))

        ckpt = {'model': model.state_dict(), 'cfg': model.cfg,
                'epoch': epoch, 'history': history}
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'v4_last.pt'))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, os.path.join(args.checkpoint_dir, 'v4_best.pt'))
        with open(os.path.join(args.checkpoint_dir, 'history.json'),
                  'w') as f:
            json.dump(history, f, indent=1)


if __name__ == '__main__':
    main()
