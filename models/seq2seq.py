from typing import List
from typing import List, Optional

import torch
from torch import nn
from torch.autograd import Variable

from modules.attention import AttentionWrapper
from modules.layers import ConvNorm
from modules.tacotron_modules import CBHG, Prenet
from options import AttentionType
from util.utils import get_mask_from_lengths


class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        # Trying smaller std
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src: torch.Tensor,
        lengths: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ):

        encoder_outputs = self.encoder(src, lengths)
        mask = get_mask_from_lengths(encoder_outputs, lengths)
        outputs, alignments = self.decoder(encoder_outputs, target, mask)

        output = {"diacritics": outputs, "attention": alignments}

        return output


class Encoder(nn.Module):
    def __init__(
        self,
        inp_vocab_size: int,
        embedding_dim: int = 512,
        layers_units: List[int] = [256, 256, 256],
        use_batch_norm: bool = False,
    ):
        super().__init__()
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
        self.layers_units = layers_units
        self.use_batch_norm = use_batch_norm

    def forward(self, inputs: torch.Tensor, inputs_lengths: torch.Tensor):

        outputs = self.embedding(inputs)

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

        return outputs

class Decoder(nn.Module):
    """A seq2seq decoder that decode a diacritic at a time ,
    Args:
    encoder_dim (int): the encoder output dim
    decoder_units (int): the number of neurons for each decoder layer
    decoder_layers (int): number of decoder layers
    """

    def __init__(
        self,
        trg_vocab_size: int,
        start_symbol_id: int,
        encoder_dim: int = 256,
        embedding_dim: int = 256,
        decoder_units: int = 256,
        decoder_layers: int = 2,
        attention_units: int = 256,
        attention_type: AttentionType = AttentionType.LocationSensitive,
        is_attention_accumulative: bool = False,
        prenet_depth: List[int] = [256, 128],
        use_prenet: bool = True,
        teacher_forcing_probability: float = 0.0,
    ):
        super().__init__()

        self.output_dim: int = trg_vocab_size
        self.start_symbol_id = start_symbol_id
        self.attention_units = attention_units
        self.decoder_units = decoder_units
        self.encoder_dim = encoder_dim
        self.use_prenet = use_prenet
        self.teacher_forcing_probability = teacher_forcing_probability
        self.is_attention_accumulative = is_attention_accumulative
        self.embbeding = nn.Embedding(trg_vocab_size, embedding_dim, padding_idx=0)
        attention_in = embedding_dim
        if use_prenet:
            self.prenet = Prenet(embedding_dim, prenet_depth)
            attention_in = prenet_depth[-1]

        self.attention_layer = nn.GRUCell(encoder_dim + attention_in, attention_units)
        self.attention_wrapper = AttentionWrapper(attention_type, attention_units)
        self.keys_layer = nn.Linear(encoder_dim, attention_units, bias=False)
        self.project_to_decoder_in = nn.Linear(
            attention_units + encoder_dim,
            decoder_units,
        )

        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(decoder_units, decoder_units) for _ in range(decoder_layers)]
        )

        self.diacritics_layer = nn.Linear(decoder_units, trg_vocab_size)
        self.device = "cuda"

    def decode(
        self,
        diacritic: torch.Tensor,
    ):
        """
        Decode one time-step
        Args:
        diacritic (Tensor): (batch_size, 1)
        Returns:
        """

        diacritic = self.embbeding(diacritic)
        if self.use_prenet:
            prenet_out = self.prenet(diacritic)
        else:
            prenet_out = diacritic

        cell_input = torch.cat((prenet_out, self.prev_attention), -1)

        self.attention_hidden = self.attention_layer(cell_input, self.attention_hidden)
        output = self.attention_hidden

        # The queries are the hidden state of the RNN layer
        attention, alignment = self.attention_wrapper(
            query=self.attention_hidden,
            values=self.encoder_outputs,
            keys=self.keys,
            mask=self.mask,
            prev_alignment=self.prev_alignment,
        )

        decoder_input = torch.cat((output, attention), -1)

        decoder_input = self.project_to_decoder_in(decoder_input)

        for idx in range(len(self.decoder_rnns)):
            self.decoder_hiddens[idx] = self.decoder_rnns[idx](
                decoder_input, self.decoder_hiddens[idx]
            )
            decoder_input = self.decoder_hiddens[idx] + decoder_input

        output = decoder_input

        output = self.diacritics_layer(output)

        if self.is_attention_accumulative:
            self.prev_alignment = self.prev_alignment + alignment
        else:
            self.prev_alignment = alignment

        self.prev_attention = attention

        return output, alignment

    def inference(self):
        """Generate diacritics one at a time"""
        batch_size = self.encoder_outputs.size(0)
        trg_len = self.encoder_outputs.size(1)
        diacritic = (
            torch.full((batch_size,), self.start_symbol_id).to(self.device).long()
        )
        outputs, alignments = [], []
        self.initialize()

        for _ in range(trg_len):
            output, alignment = self.decode(diacritic=diacritic)

            outputs.append(output)
            alignments.append(alignment)
            diacritic = torch.max(output, 1).indices

        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        return outputs, alignments

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        diacritics: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
    ):
        """calculate  forward propagation
        Args:
        encoder_outputs (Tensor): the output of the encoder
        (batch_size, Tx, encoder_units * 2)
        diacritics(Tensor): target sequence
        input_mask (Tensor):  the inputs mask (batch_size, Tx)
        """
        self.mask = input_mask
        self.encoder_outputs = encoder_outputs
        self.keys = self.keys_layer(encoder_outputs)

        if diacritics is None:
            return self.inference()

        batch_size = diacritics.size(0)
        trg_len = diacritics.size(1)

        # Init decoder states
        outputs = []
        alignments = []

        self.initialize()

        diacritic = (
            torch.full((batch_size,), self.start_symbol_id).to(self.device).long()
        )

        for time in range(trg_len):
            output, alignment = self.decode(diacritic=diacritic)
            outputs += [output]
            alignments += [alignment]
            #if random.random() > self.teacher_forcing_probability:
            diacritic = diacritics[:, time]  # use training input
            #else:
                #diacritic = torch.max(output, 1).indices  # use last output

        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments

    def initialize(self):
        """Initialize the first step variables"""
        batch_size = self.encoder_outputs.size(0)
        src_len = self.encoder_outputs.size(1)
        self.attention_hidden = Variable(
            torch.zeros(batch_size, self.attention_units)
        ).to(self.device)
        self.decoder_hiddens = [
            Variable(torch.zeros(batch_size, self.decoder_units)).to(self.device)
            for _ in range(len(self.decoder_rnns))
        ]
        self.prev_attention = Variable(torch.zeros(batch_size, self.encoder_dim)).to(
            self.device
        )
        self.prev_alignment = Variable(torch.zeros(batch_size, src_len)).to(self.device)
