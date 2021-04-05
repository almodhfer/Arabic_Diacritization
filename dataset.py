"""
Loading the diacritization dataset
"""

import os

from diacritization_evaluation import util
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config_manager import ConfigManager


class DiacritizationDataset(Dataset):
    """
    The diacritization dataset
    """

    def __init__(self, config_manager: ConfigManager, list_ids, data):
        "Initialization"
        self.list_ids = list_ids
        self.data = data
        self.text_encoder = config_manager.text_encoder
        self.config = config_manager.config

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_ids)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        id = self.list_ids[index]
        if self.config["is_data_preprocessed"]:
            data = self.data.iloc[id]
            inputs = torch.Tensor(self.text_encoder.input_to_sequence(data[1]))
            targets = torch.Tensor(
                self.text_encoder.target_to_sequence(
                    data[2].split(self.config["diacritics_separator"])
                )
            )
            return inputs, targets, data[0]

        data = self.data[id]
        data = self.text_encoder.clean(data)

        text, inputs, diacritics = util.extract_haraqat(data)
        inputs = torch.Tensor(self.text_encoder.input_to_sequence("".join(inputs)))
        diacritics = torch.Tensor(self.text_encoder.target_to_sequence(diacritics))

        return inputs, diacritics, text


def collate_fn(data):
    """
    Padding the input and output sequences
    """

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    src_seqs, trg_seqs, original = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    batch = {
        "original": original,
        "src": src_seqs,
        "target": trg_seqs,
        "lengths": torch.LongTensor(src_lengths),  # src_lengths = trg_lengths
    }
    return batch


def load_training_data(config_manager: ConfigManager, loader_parameters):
    """
    Loading the training data using pandas
    """

    if not config_manager.config["load_training_data"]:
        return []

    path = os.path.join(config_manager.data_dir, "train.csv")
    if config_manager.config["is_data_preprocessed"]:
        train_data = pd.read_csv(
            path,
            encoding="utf-8",
            sep=config_manager.config["data_separator"],
            nrows=config_manager.config["n_training_examples"],
            header=None,
        )

        # train_data = train_data[train_data[0] <= config_manager.config["max_len"]]
        training_set = DiacritizationDataset(
            config_manager, train_data.index, train_data
        )
    else:
        with open(path, encoding="utf8") as file:
            train_data = file.readlines()
            train_data = [
                text
                for text in train_data
                if len(text) <= config_manager.config["max_len"]
            ]
        training_set = DiacritizationDataset(
            config_manager, [idx for idx in range(len(train_data))], train_data
        )

    train_iterator = DataLoader(
        training_set, collate_fn=collate_fn, **loader_parameters
    )

    print(f"Length of training iterator = {len(train_iterator)}")
    return train_iterator


def load_test_data(config_manager: ConfigManager, loader_parameters):
    """
    Loading the test data using pandas
    """
    if not config_manager.config["load_test_data"]:
        return []
    test_file_name = config_manager.config.get("test_file_name", "test.csv")
    path = os.path.join(config_manager.data_dir, test_file_name)
    if config_manager.config["is_data_preprocessed"]:
        test_data = pd.read_csv(
            path,
            encoding="utf-8",
            sep=config_manager.config["data_separator"],
            nrows=config_manager.config["n_test_examples"],
            header=None,
        )
        # test_data = test_data[test_data[0] <= config_manager.config["max_len"]]
        test_dataset = DiacritizationDataset(config_manager, test_data.index, test_data)
    else:
        with open(path, encoding="utf8") as file:
            test_data = file.readlines()
        test_data = [
            text for text in test_data if len(text) <= config_manager.config["max_len"]
        ]
        test_dataset = DiacritizationDataset(
            config_manager, [idx for idx in range(len(test_data))], test_data
        )

    test_iterator = DataLoader(test_dataset, collate_fn=collate_fn, **loader_parameters)

    print(f"Length of test iterator = {len(test_iterator)}")
    return test_iterator


def load_validation_data(config_manager: ConfigManager, loader_parameters):
    """
    Loading the validation data using pandas
    """

    if not config_manager.config["load_validation_data"]:
        return []
    path = os.path.join(config_manager.data_dir, "eval.csv")
    if config_manager.config["is_data_preprocessed"]:
        valid_data = pd.read_csv(
            path,
            encoding="utf-8",
            sep=config_manager.config["data_separator"],
            nrows=config_manager.config["n_validation_examples"],
            header=None,
        )

        # valid_data = valid_data[valid_data[0] <= config_manager.config["max_len"]]
        valid_dataset = DiacritizationDataset(
            config_manager, valid_data.index, valid_data
        )
    else:
        with open(path, encoding="utf8") as file:
            valid_data = file.readlines()

        valid_data = [
            text for text in valid_data if len(text) <= config_manager.config["max_len"]
        ]
        valid_dataset = DiacritizationDataset(
            config_manager, [idx for idx in range(len(valid_data))], valid_data
        )

    valid_iterator = DataLoader(
        valid_dataset, collate_fn=collate_fn, **loader_parameters
    )

    print(f"Length of valid iterator = {len(valid_iterator)}")
    return valid_iterator


def load_iterators(config_manager: ConfigManager):
    """
    Load the data iterators
    Args:
    """
    params = {
        "batch_size": config_manager.config["batch_size"],
        "shuffle": True,
        "num_workers": 2,
    }
    train_iterator = load_training_data(config_manager, loader_parameters=params)
    valid_iterator = load_validation_data(config_manager, loader_parameters=params)
    test_iterator = load_test_data(config_manager, loader_parameters=params)
    return train_iterator, test_iterator, valid_iterator
