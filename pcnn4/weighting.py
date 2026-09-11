"""V*-relative sample weights.

Let x = V* of the player's chosen solve, v = V* of a candidate solve, and
d = v - x.

- Inclusion gate: solves with d <= -v_window are dropped; the gate rises
  smoothly (smoothstep) from 0 at d = -v_window to 1 at d = 0, so
  "somewhat okay but slightly worse" saves count with reduced weight.
- Skip boost: a solve the player did NOT choose despite a strictly better
  save is strong negative evidence; the weight rises from 1x at d = 0 to
  max_boost (10x) at d >= d_significant, exponentially interpolated
  (max_boost ** smoothstep(d / d_significant)).
- Positives: the chosen solve's samples share a total weight of
  pos_ratio * (total negative weight in the situation), floored per sample,
  so the single positive solve is not drowned by its many negatives.

Note: on the real V* table (range ~3842-4353) candidate spreads within one
situation are often < 0.01, so most negatives sit near weight 1.0; the
boost only fires when a genuinely better save was skipped.
"""
from pcnn4.config import WEIGHT_CONFIG


def _cfg(cfg):
    return {**WEIGHT_CONFIG, **(cfg or {})}


def smoothstep(t):
    """0 for t<=0, 1 for t>=1, 3t^2-2t^3 between."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def gate(d, cfg=None):
    c = _cfg(cfg)
    return smoothstep((d + c['v_window']) / c['v_window'])


def boost(d, cfg=None):
    c = _cfg(cfg)
    return c['max_boost'] ** smoothstep(d / c['d_significant'])


def include(d, cfg=None):
    """Whether a candidate solve enters the dataset at all."""
    return d > -_cfg(cfg)['v_window']


def negative_weight(d, cfg=None):
    """Weight of a not-chosen solve at V* difference d."""
    return gate(d, cfg) * boost(d, cfg)


def positive_weight_each(n_pos, neg_total, cfg=None):
    """Per-sample weight for the chosen solve's n_pos order samples."""
    c = _cfg(cfg)
    if n_pos <= 0:
        raise ValueError("n_pos must be positive")
    return max(c['pos_floor'], c['pos_ratio'] * neg_total / n_pos)
