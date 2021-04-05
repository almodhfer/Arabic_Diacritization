from typing import List
from models.seq2seq import Seq2Seq, Decoder as Seq2SeqDecoder
from modules.tacotron_modules import CBHG, Prenet
from torch import nn


class Tacotron(Seq2Seq):
    pass


class Encoder(nn.Module):
    def __init__(
        self,
        inp_vocab_size: int,
        embedding_dim: int = 512,
        use_prenet: bool = True,
        prenet_sizes: List[int] = [256, 128],
        cbhg_gru_units: int = 128,
        cbhg_filters: int = 16,
        cbhg_projections: List[int] = [128, 128],
        padding_idx: int = 0,
    ):
        super().__init__()
        self.use_prenet = use_prenet

        self.embedding = nn.Embedding(
            inp_vocab_size, embedding_dim, padding_idx=padding_idx
        )
        if use_prenet:
            self.prenet = Prenet(embedding_dim, prenet_depth=prenet_sizes)
        self.cbhg = CBHG(
            prenet_sizes[-1] if use_prenet else embedding_dim,
            cbhg_gru_units,
            K=cbhg_filters,
            projections=cbhg_projections,
        )

    def forward(self, inputs, input_lengths=None):

        outputs = self.embedding(inputs)
        if self.use_prenet:
            outputs = self.prenet(outputs)
        return self.cbhg(outputs, input_lengths)


class Decoder(Seq2SeqDecoder):
    pass
