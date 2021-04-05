"""
The CBHG model implementation
"""
from typing import List, Optional

from torch import nn
import torch

from modules.tacotron_modules import CBHG, Prenet


class CBHGModel(nn.Module):
    """CBHG model implementation as described in the paper:
     https://ieeexplore.ieee.org/document/9274427

    Args:
    inp_vocab_size (int): the number of the input symbols
    targ_vocab_size (int): the number of the target symbols (diacritics)
    embedding_dim (int): the embedding  size
    use_prenet (bool): whether to use prenet or not
    prenet_sizes (List[int]): the sizes of the prenet networks
    cbhg_gru_units (int): the number of units of the CBHG GRU, which is the last
    layer of the CBHG Model.
    cbhg_filters (int): number of filters used in the CBHG module
    cbhg_projections: projections used in the CBHG module

    Returns:
    diacritics Dict[str, Tensor]:
    """

    def __init__(
        self,
        inp_vocab_size: int,
        targ_vocab_size: int,
        embedding_dim: int = 512,
        use_prenet: bool = True,
        prenet_sizes: List[int] = [512, 256],
        cbhg_gru_units: int = 512,
        cbhg_filters: int = 16,
        cbhg_projections: List[int] = [128, 256],
        post_cbhg_layers_units: List[int] = [256, 256],
        post_cbhg_use_batch_norm: bool = True
    ):
        super().__init__()
        self.use_prenet = use_prenet
        self.embedding = nn.Embedding(inp_vocab_size, embedding_dim)
        if self.use_prenet:
            self.prenet = Prenet(embedding_dim, prenet_depth=prenet_sizes)

        self.cbhg = CBHG(
            prenet_sizes[-1] if self.use_prenet else embedding_dim,
            cbhg_gru_units,
            K=cbhg_filters,
            projections=cbhg_projections,
        )

        layers = []
        post_cbhg_layers_units = [cbhg_gru_units] + post_cbhg_layers_units

        for i in range(1, len(post_cbhg_layers_units)):
            layers.append(
                nn.LSTM(
                    post_cbhg_layers_units[i - 1] * 2,
                    post_cbhg_layers_units[i],
                    bidirectional=True,
                    batch_first=True,
                )
            )
            if post_cbhg_use_batch_norm:
                layers.append(nn.BatchNorm1d(post_cbhg_layers_units[i] * 2))

        self.post_cbhg_layers = nn.ModuleList(layers)
        self.projections = nn.Linear(post_cbhg_layers_units[-1] * 2, targ_vocab_size)
        self.post_cbhg_layers_units = post_cbhg_layers_units
        self.post_cbhg_use_batch_norm = post_cbhg_use_batch_norm


    def forward(
        self,
        src: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,  # not required in this model
    ):
        """Compute forward propagation"""

        # src = [batch_size, src len]
        # lengths = [batch_size]
        # target = [batch_size, trg len]

        embedding_out = self.embedding(src)
        # embedding_out; [batch_size, src_len, embedding_dim]

        cbhg_input = embedding_out
        if self.use_prenet:
            cbhg_input = self.prenet(embedding_out)

            # cbhg_input = [batch_size, src_len, prenet_sizes[-1]]

        outputs = self.cbhg(cbhg_input, lengths)

        hn = torch.zeros((2, 2, 2))
        cn = torch.zeros((2, 2, 2))

        for i, layer in enumerate(self.post_cbhg_layers):
            if isinstance(layer, nn.BatchNorm1d):
                outputs = layer(outputs.permute(0, 2, 1))
                outputs = outputs.permute(0, 2, 1)
                continue
            if i > 0:
                outputs, (hn, cn) = layer(outputs, (hn, cn))
            else:
                outputs, (hn, cn) = layer(outputs)


        predictions = self.projections(outputs)

        # predictions = [batch_size, src len, targ_vocab_size]

        output = {"diacritics": predictions}

        return output
