import torch

from pcnn4.config import COLOR_TO_IDX, NUM_COLORS
from pcnn4.embeddings import MirrorColorEmbedding, OrdinalEmbedding


def test_color_weight_shape():
    emb = MirrorColorEmbedding(dim=8)
    w = emb.weight()
    assert w.shape == (NUM_COLORS, 8)


def test_mirror_structure_enforced():
    emb = MirrorColorEmbedding(dim=8)
    w = emb.weight()
    l, j = w[COLOR_TO_IDX['L']], w[COLOR_TO_IDX['J']]
    s, z = w[COLOR_TO_IDX['S']], w[COLOR_TO_IDX['Z']]
    # L and J are reflections around a shared base; same for S and Z
    assert torch.allclose(l + j, 2 * emb.base[0])
    assert torch.allclose(s + z, 2 * emb.base[1])
    assert torch.allclose(l - j, 2 * emb.diff[0])


def test_mirror_structure_survives_training_step():
    emb = MirrorColorEmbedding(dim=8)
    opt = torch.optim.SGD(emb.parameters(), lr=0.1)
    idx = torch.randint(0, NUM_COLORS, (16, 4, 10))
    loss = emb(idx).pow(2).sum()
    loss.backward()
    opt.step()
    w = emb.weight()
    assert torch.allclose(w[COLOR_TO_IDX['L']] + w[COLOR_TO_IDX['J']],
                          2 * emb.base[0])
    assert torch.allclose(w[COLOR_TO_IDX['S']] + w[COLOR_TO_IDX['Z']],
                          2 * emb.base[1])


def test_lookup_matches_weight_rows():
    emb = MirrorColorEmbedding(dim=8)
    idx = torch.tensor([[COLOR_TO_IDX['N'], COLOR_TO_IDX['T']]])
    out = emb(idx)
    assert out.shape == (1, 2, 8)
    assert torch.allclose(out[0, 0], emb.weight()[0])
    assert torch.allclose(out[0, 1], emb.weight()[1])


def test_gradients_flow():
    emb = MirrorColorEmbedding(dim=8)
    idx = torch.tensor([[COLOR_TO_IDX['L'], COLOR_TO_IDX['S']]])
    emb(idx).sum().backward()
    assert emb.base.grad is not None and emb.base.grad.abs().sum() > 0
    assert emb.diff.grad is not None


def test_ordinal_embedding():
    emb = OrdinalEmbedding(dim=8)
    idx = torch.randint(0, 11, (2, 4, 10))
    assert emb(idx).shape == (2, 4, 10, 8)
