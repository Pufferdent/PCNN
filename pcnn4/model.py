"""PCChoiceNet: (broken-board colors, placement ordinals) -> one probability.

Each of the 40 cells is a token: color embedding (8) + ordinal embedding (8)
concatenated to 16 channels on the 4x10 grid, a small conv stack, then an
MLP head to a single logit. ``forward`` returns logits (use
BCEWithLogitsLoss); ``probability`` applies the sigmoid.
"""
import torch
import torch.nn as nn

from pcnn4.config import MODEL_CONFIG, BOARD_H, BOARD_W
from pcnn4.embeddings import MirrorColorEmbedding, OrdinalEmbedding


class PCChoiceNet(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = {**MODEL_CONFIG, **(cfg or {})}
        self.cfg = cfg
        self.color_emb = MirrorColorEmbedding(cfg['color_embed_dim'])
        self.ordinal_emb = OrdinalEmbedding(cfg['ordinal_embed_dim'])

        in_ch = cfg['color_embed_dim'] + cfg['ordinal_embed_dim']
        convs = []
        k = cfg['conv_kernel']
        for out_ch in cfg['conv_channels']:
            convs.append(nn.Conv2d(in_ch, out_ch, k, padding=k // 2))
            convs.append(nn.ReLU())
            in_ch = out_ch
        self.convs = nn.Sequential(*convs)

        head = [nn.Flatten()]
        dim = in_ch * BOARD_H * BOARD_W
        for hidden in cfg['head_hidden']:
            head.append(nn.Linear(dim, hidden))
            head.append(nn.ReLU())
            head.append(nn.Dropout(cfg['dropout']))
            dim = hidden
        head.append(nn.Linear(dim, 1))
        self.head = nn.Sequential(*head)

    def forward(self, color_idx, ordinal_idx):
        """color_idx, ordinal_idx: LongTensor (B, 4, 10). Returns (B,) logits."""
        x = torch.cat([self.color_emb(color_idx),
                       self.ordinal_emb(ordinal_idx)], dim=-1)
        x = x.permute(0, 3, 1, 2)          # (B, C, H, W)
        x = self.convs(x)
        return self.head(x).squeeze(-1)

    def probability(self, color_idx, ordinal_idx):
        return torch.sigmoid(self.forward(color_idx, ordinal_idx))
