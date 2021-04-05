from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from options import AttentionType


class BahdanauAttention(nn.Module):
    def __init__(self, dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        """
        Args:
            query: (B, 1, dim) or (batch, dim)
            processed_memory: (batch, max_time, dim)
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        # (batch, 1, dim)
        query = self.query_layer(query)

        # (batch, max_time, 1)
        alignment = self.v(self.tanh(query + keys))

        # (batch, max_time)
        return alignment.squeeze(-1)


class LocationSensitive(nn.Module):
    def __init__(self, dim):
        super(LocationSensitive, self).__init__()
        self.query_layer = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, 1, bias=True)
        self.location_layer = nn.Linear(32, dim, bias=False)
        padding = int((31 - 1) / 2)
        self.location_conv = torch.nn.Conv1d(
            1, 32, kernel_size=31, stride=1, padding=padding, dilation=1, bias=False
        )

        self.score_mask_value = -float("inf")

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        prev_alignments: torch.Tensor,
    ):
        # keys = keys.permute(1,0,2)
        query = self.query_layer(query)
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
            # -> [batch_size, 1, attention_dim]

        alignments = prev_alignments.unsqueeze(1)

        # location features [batch_size, max_time, filters]
        filters = self.location_conv(alignments)
        location_features = self.location_layer(filters.transpose(1, 2))

        alignments = self.v(torch.tanh(query + location_features + keys))
        return alignments.squeeze(-1)


class AttentionWrapper(nn.Module):
    def __init__(
        self,
        attention_type: AttentionType = AttentionType.LocationSensitive,
        attention_units: int = 256,
        score_mask_value=-float("inf"),
    ):
        super().__init__()
        self.score_mask_value = score_mask_value
        self.attention_type = attention_type

        if attention_type == AttentionType.LocationSensitive:
            self.attention_mechanism = LocationSensitive(attention_units)
        elif attention_type == AttentionType.Content_Based:
            self.attention_mechanism = BahdanauAttention(attention_units)
        else:
            raise Exception("The attention type is not known")

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        prev_alignment: Optional[torch.Tensor] = None,
    ):

        # Alignment
        # (batch, max_time)
        if self.attention_type == AttentionType.Content_Based:
            alignment = self.attention_mechanism(query, keys)
        else:
            alignment = self.attention_mechanism(query, keys, prev_alignment)

        # Attention context vector

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        alignment = F.softmax(alignment, dim=1)
        attention = torch.bmm(alignment.unsqueeze(1), values)
        attention = attention.squeeze(1)

        return attention, alignment


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim * 2, hid_dim)

        if dropout != 0.0:
            self.dropout = nn.Dropout(dropout)

        self.use_dropout = dropout != 0.0

        device = next(self.parameters()).device

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -float("inf"))

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        if self.use_dropout:
            context_vector = torch.matmul(self.dropout(attention), V)
        else:
            context_vector = torch.matmul(attention, V)

        # x = [batch size, n heads, query len, head dim]

        context_vector = context_vector.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        context_vector = context_vector.view(batch_size, -1, self.hid_dim)

        x = torch.cat((query, context_vector), dim=-1)

        # x = [batch size, query len, hid dim * 2]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention
