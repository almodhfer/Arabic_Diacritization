import os
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from itertools import repeat
from util.decorators import ignore_exception
from dataclasses import dataclass
import numpy as np


@dataclass
class ErrorRate:
    wer: float
    der: float
    wer_without_case_ending: float
    der_without_case_ending: float


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


@ignore_exception
def plot_alignment(alignment: torch.Tensor, path: str, global_step: Any = 0):
    """
    Plot alignment and save it into a path
    Args:
    alignment (Tensor): the encoder-decoder alignment
    path (str): a path used to save the alignment plot
    global_step (int): used in the name of the output alignment plot
    """
    alignment = alignment.squeeze(1).transpose(0, 1).cpu().detach().numpy()
    fig, axs = plt.subplots()
    img = axs.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(img, ax=axs)
    xlabel = "Decoder timestep"
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()
    plot_name = f"{global_step}.png"
    plt.savefig(os.path.join(path, plot_name), dpi=300, format="png")
    plt.close()


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    mask = memory.data.new(memory.size(0), memory.size(1)).bool().zero_()
    for idx, length in enumerate(memory_lengths):
        mask[idx][:length] = 1
    return ~mask


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def get_encoder_layers_attentions(model):
    attentions = []
    for layer in model.encoder.layers:
        attentions.append(layer.self_attention.attention)
    return attentions


def get_decoder_layers_attentions(model):
    self_attns, src_attens = [], []
    for layer in model.decoder.layers:
        self_attns.append(layer.self_attention.attention)
        src_attens.append(layer.encoder_attention.attention)
    return self_attns, src_attens


def display_attention(
    attention, path, global_step: int, name="att", n_heads=4, n_rows=2, n_cols=2
):
    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 15))

    for i in range(n_heads):

        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].transpose(0, 1).cpu().detach().numpy()
        cax = ax.imshow(_attention, aspect="auto", origin="lower", interpolation="none")

    plot_name = f"{global_step}-{name}.png"
    plt.savefig(os.path.join(path, plot_name), dpi=300, format="png")
    plt.close()


def plot_multi_head(model, path, global_step):
    encoder_attentions = get_encoder_layers_attentions(model)
    decoder_attentions, attentions = get_decoder_layers_attentions(model)
    for i in range(len(attentions)):
        display_attention(
            attentions[0][0], path, global_step, f"encoder-decoder-layer{i + 1}"
        )
    for i in range(len(decoder_attentions)):
        display_attention(
            decoder_attentions[0][0], path, global_step, f"decoder-layer{i + 1}"
        )
    for i in range(len(encoder_attentions)):
        display_attention(
            encoder_attentions[0][0], path, global_step, f"encoder-layer {i + 1}"
        )


def make_src_mask(src, pad_idx=0):

    # src = [batch size, src len]

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # src_mask = [batch size, 1, 1, src len]

    return src_mask


def get_angles(pos, i, model_dim):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(model_dim))
    return pos * angle_rates


def positional_encoding(position, model_dim):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(model_dim)[np.newaxis, :],
        model_dim,
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.from_numpy(pos_encoding)


def calculate_error_rates(original_file_path: str, target_file_path: str) -> ErrorRate:
    """
    Calculates ErrorRates from paths
    """
    assert os.path.isfile(original_file_path)
    assert os.path.isfile(target_file_path)

    _wer = wer.calculate_wer_from_path(
        inp_path=original_file_path, out_path=target_file_path, case_ending=True
    )

    _wer_without_case_ending = wer.calculate_wer_from_path(
        inp_path=original_file_path, out_path=target_file_path, case_ending=False
    )

    _der = der.calculate_der_from_path(
        inp_path=original_file_path, out_path=target_file_path, case_ending=True
    )

    _der_without_case_ending = der.calculate_der_from_path(
        inp_path=original_file_path, out_path=target_file_path, case_ending=False
    )

    error_rates = ErrorRate(
        _wer,
        _der,
        _wer_without_case_ending,
        _der_without_case_ending,
    )

    return error_rates


def categorical_accuracy(preds, y, tag_pad_idx, device="cuda"):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(
        dim=1, keepdim=True
    )  # get the index of the max probability
    non_pad_elements = torch.nonzero((y != tag_pad_idx))
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)


def write_to_files(input_path, output_path, input_list, output_list):
    with open(input_path, "w", encoding="utf8") as file:
        for inp in input_list:
            file.write(inp + "\n")
    with open(output_path, "w", encoding="utf8") as file:
        for out in output_list:
            file.write(out + "\n")


def make_src_mask(src: torch.Tensor, pad_idx=0):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_trg_mask(trg, trg_pad_idx=0):

    # trg = [batch size, trg len]

    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)

    # trg_pad_mask = [batch size, 1, 1, trg len]

    trg_len = trg.shape[1]

    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()

    # trg_sub_mask = [trg len, trg len]

    trg_mask = trg_pad_mask & trg_sub_mask

    # trg_mask = [batch size, 1, trg len, trg len]

    return trg_mask
