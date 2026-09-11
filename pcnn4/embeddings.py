"""Cell embeddings with hard mirror structure.

The horizontal mirror of a PC solve is equally valid: L<->J and S<->Z swap
while T/I/O map to themselves. v3 only *initialized* its piece embedding
with that structure; v4 enforces it for the whole of training by
parametrizing L = b_LJ + d_LJ, J = b_LJ - d_LJ (and likewise S/Z), so
mirrored pieces stay exact reflections of each other in embedding space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcnn4.config import ORDINAL_VOCAB


class MirrorColorEmbedding(nn.Module):
    """8-dim embedding over {empty, T, I, L, J, S, Z, O} with L/J and S/Z
    tied as base +/- diff pairs. Row order matches COLOR_CHARS:
    N, T, I, L, J, S, Z, O."""

    def __init__(self, dim=8, init_scale=0.1):
        super().__init__()
        self.dim = dim
        self.free = nn.Parameter(torch.randn(4, dim) * init_scale)  # N T I O
        self.base = nn.Parameter(torch.randn(2, dim) * init_scale)  # LJ, SZ
        self.diff = nn.Parameter(torch.randn(2, dim) * init_scale)  # LJ, SZ

    def weight(self):
        n, t, i, o = self.free
        l = self.base[0] + self.diff[0]
        j = self.base[0] - self.diff[0]
        s = self.base[1] + self.diff[1]
        z = self.base[1] - self.diff[1]
        return torch.stack([n, t, i, l, j, s, z, o])

    def forward(self, idx):
        return F.embedding(idx, self.weight())


class OrdinalEmbedding(nn.Module):
    """8-dim embedding of the placement ordinal (0 = empty cell, 1..10 = the
    step at which the piece covering this cell was placed)."""

    def __init__(self, dim=8):
        super().__init__()
        self.emb = nn.Embedding(ORDINAL_VOCAB, dim)

    def forward(self, idx):
        return self.emb(idx)
