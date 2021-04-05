from typing import List
from torch import nn
import torch


class BaseLineModel(nn.Module):
    def __init__(
        self,
        inp_vocab_size: int,
        targ_vocab_size: int,
        embedding_dim: int = 512,
        layers_units: List[int] = [256, 256, 256],
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.targ_vocab_size = targ_vocab_size
        self.embedding = nn.Embedding(inp_vocab_size, embedding_dim)

        layers_units = [embedding_dim // 2] + layers_units

        layers = []

        for i in range(1, len(layers_units)):
            layers.append(
                nn.LSTM(
                    layers_units[i - 1] * 2,
                    layers_units[i],
                    bidirectional=True,
                    batch_first=True,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layers_units[i] * 2))

        self.layers = nn.ModuleList(layers)
        self.projections = nn.Linear(layers_units[-1] * 2, targ_vocab_size)
        self.layers_units = layers_units
        self.use_batch_norm = use_batch_norm

    def forward(self, src: torch.Tensor, lengths: torch.Tensor, target=None):

        outputs = self.embedding(src)

        # embedded_inputs = [batch_size, src_len, embedding_dim]

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.BatchNorm1d):
                outputs = layer(outputs.permute(0, 2, 1))
                outputs = outputs.permute(0, 2, 1)
                continue
            if i > 0:
                outputs, (hn, cn) = layer(outputs, (hn, cn))
            else:
                outputs, (hn, cn) = layer(outputs)

        predictions = self.projections(outputs)

        output = {"diacritics": predictions}

        return output
