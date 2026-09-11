import torch

from pcnn4.config import NUM_COLORS, ORDINAL_VOCAB
from pcnn4.model import PCChoiceNet


def _inputs(b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    color = torch.randint(0, NUM_COLORS, (b, 4, 10), generator=g)
    ordinal = torch.randint(0, ORDINAL_VOCAB, (b, 4, 10), generator=g)
    return color, ordinal


def test_forward_shape():
    model = PCChoiceNet()
    color, ordinal = _inputs(3)
    logits = model(color, ordinal)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_probability_range():
    model = PCChoiceNet()
    color, ordinal = _inputs(5)
    p = model.probability(color, ordinal)
    assert ((p > 0) & (p < 1)).all()


def test_single_scalar_output():
    model = PCChoiceNet()
    color, ordinal = _inputs(1)
    assert model(color, ordinal).numel() == 1


def test_backward_and_param_count():
    model = PCChoiceNet()
    color, ordinal = _inputs(4)
    label = torch.tensor([1.0, 0.0, 0.0, 1.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(color, ordinal), label)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    n_params = sum(p.numel() for p in model.parameters())
    assert 100_000 < n_params < 10_000_000
