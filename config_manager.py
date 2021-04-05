from enum import Enum
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict

import ruamel.yaml
import torch

from models.baseline import BaseLineModel
from models.cbhg import CBHGModel
from models.seq2seq import Decoder as Seq2SeqDecoder, Encoder as Seq2SeqEncoder, Seq2Seq
from models.tacotron_based import (
    Decoder as TacotronDecoder,
    Encoder as TacotronEncoder,
    Tacotron,
)

from options import AttentionType, LossType, OptimizerType
from util.text_encoders import (
    ArabicEncoderWithStartSymbol,
    BasicArabicEncoder,
    TextEncoder,
)


class ConfigManager:
    """Co/home/almodhfer/Projects/daicritization/temp_results/CA_MSA/cbhg-new/model-10.ptnfig Manager"""

    def __init__(self, config_path: str, model_kind: str):
        available_models = [
            "baseline",
            "cbhg",
            "seq2seq",
            "tacotron_based",
        ]
        if model_kind not in available_models:
            raise TypeError(f"model_kind must be in {available_models}")
        self.config_path = Path(config_path)
        self.model_kind = model_kind
        self.yaml = ruamel.yaml.YAML()
        self.config: Dict[str, Any] = self._load_config()
        self.git_hash = self._get_git_hash()
        self.session_name = ".".join(
            [
                self.config["data_type"],
                self.config["session_name"],
                f"{model_kind}",
            ]
        )

        self.data_dir = Path(
            os.path.join(self.config["data_directory"], self.config["data_type"])
        )
        self.base_dir = Path(
            os.path.join(self.config["log_directory"], self.session_name)
        )
        self.log_dir = Path(os.path.join(self.base_dir, "logs"))
        self.prediction_dir = Path(os.path.join(self.base_dir, "predictions"))
        self.plot_dir = Path(os.path.join(self.base_dir, "plots"))
        self.models_dir = Path(os.path.join(self.base_dir, "models"))
        self.text_encoder: TextEncoder = self.get_text_encoder()
        self.config["len_input_symbols"] = len(self.text_encoder.input_symbols)
        self.config["len_target_symbols"] = len(self.text_encoder.target_symbols)
        if self.model_kind in ["seq2seq", "tacotron_based"]:
            self.config["attention_type"] = AttentionType[self.config["attention_type"]]
        self.config["optimizer"] = OptimizerType[self.config["optimizer_type"]]

    def _load_config(self):
        with open(self.config_path, "rb") as model_yaml:
            _config = self.yaml.load(model_yaml)
        return _config

    @staticmethod
    def _get_git_hash():
        try:
            return (
                subprocess.check_output(["git", "describe", "--always"])
                .strip()
                .decode()
            )
        except Exception as e:
            print(f"WARNING: could not retrieve git hash. {e}")

    def _check_hash(self):
        try:
            git_hash = (
                subprocess.check_output(["git", "describe", "--always"])
                .strip()
                .decode()
            )
            if self.config["git_hash"] != git_hash:
                print(
                    f"""WARNING: git hash mismatch. Current: {git_hash}.
                    Config hash: {self.config['git_hash']}"""
                )
        except Exception as e:
            print(f"WARNING: could not check git hash. {e}")

    @staticmethod
    def _print_dict_values(values, key_name, level=0, tab_size=2):
        tab = level * tab_size * " "
        print(tab + "-", key_name, ":", values)

    def _print_dictionary(self, dictionary, recursion_level=0):
        for key in dictionary.keys():
            if isinstance(key, dict):
                recursion_level += 1
                self._print_dictionary(dictionary[key], recursion_level)
            else:
                self._print_dict_values(
                    dictionary[key], key_name=key, level=recursion_level
                )

    def print_config(self):
        print("\nCONFIGURATION", self.session_name)
        self._print_dictionary(self.config)

    def update_config(self):
        self.config["git_hash"] = self._get_git_hash()

    def dump_config(self):
        self.update_config()
        _config = {}
        for key, val in self.config.items():
            if isinstance(val, Enum):
                _config[key] = val.name
            else:
                _config[key] = val
        with open(self.base_dir / "config.yml", "w") as model_yaml:
            self.yaml.dump(_config, model_yaml)

    def create_remove_dirs(
        self,
        clear_dir: bool = False,
        clear_logs: bool = False,
        clear_weights: bool = False,
        clear_all: bool = False,
    ):
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.plot_dir.mkdir(exist_ok=True)
        self.prediction_dir.mkdir(exist_ok=True)
        if clear_dir:
            delete = input(f"Delete {self.log_dir} AND {self.models_dir}? (y/[n])")
            if delete == "y":
                shutil.rmtree(self.log_dir, ignore_errors=True)
                shutil.rmtree(self.models_dir, ignore_errors=True)
        if clear_logs:
            delete = input(f"Delete {self.log_dir}? (y/[n])")
            if delete == "y":
                shutil.rmtree(self.log_dir, ignore_errors=True)
        if clear_weights:
            delete = input(f"Delete {self.models_dir}? (y/[n])")
            if delete == "y":
                shutil.rmtree(self.models_dir, ignore_errors=True)
        self.log_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

    def get_last_model_path(self):
        """
        Given a checkpoint, get the last save model name
        Args:
        checkpoint (str): the path where models are saved
        """
        models = os.listdir(self.models_dir)
        models = [model for model in models if model[-3:] == ".pt"]
        if len(models) == 0:
            return None
        _max = max(int(m.split(".")[0].split("-")[0]) for m in models)
        model_name = f"{_max}-snapshot.pt"
        last_model_path = os.path.join(self.models_dir, model_name)

        return last_model_path

    def load_model(self, model_path: str = None):
        """
        loading a model from path
        Args:
        checkpoint (str): the path to the model
        name (str): the name of the model, which is in the path
        model (Tacotron): the model  to load its save state
        optimizer: the optimizer to load its saved state
        """

        model = self.get_model()

        with open(self.base_dir / f"{self.model_kind}_network.txt", "w") as file:
            file.write(str(model))

        if model_path is None:
            last_model_path = self.get_last_model_path()
            if last_model_path is None:
                return model, 1
        else:
            last_model_path = model_path

        saved_model = torch.load(last_model_path)
        out = model.load_state_dict(saved_model["model_state_dict"])
        print(out)
        global_step = saved_model["global_step"] + 1
        return model, global_step

    def get_model(self, ignore_hash=False):
        if not ignore_hash:
            self._check_hash()
        if self.model_kind == "cbhg":
            return self.get_cbhg()

        elif self.model_kind == "seq2seq":
            return self.get_seq2seq()

        elif self.model_kind == "tacotron_based":
            return self.get_tacotron_based()

        elif self.model_kind == "baseline":
            return self.get_baseline()

    def get_baseline(self):
        model = BaseLineModel(
            embedding_dim=self.config["embedding_dim"],
            inp_vocab_size=self.config["len_input_symbols"],
            targ_vocab_size=self.config["len_target_symbols"],
            layers_units=self.config["layers_units"],
            use_batch_norm=self.config["use_batch_norm"],
        )

        return model

    def get_cbhg(self):
        model = CBHGModel(
            embedding_dim=self.config["embedding_dim"],
            inp_vocab_size=self.config["len_input_symbols"],
            targ_vocab_size=self.config["len_target_symbols"],
            use_prenet=self.config["use_prenet"],
            prenet_sizes=self.config["prenet_sizes"],
            cbhg_gru_units=self.config["cbhg_gru_units"],
            cbhg_filters=self.config["cbhg_filters"],
            cbhg_projections=self.config["cbhg_projections"],
            post_cbhg_layers_units=self.config["post_cbhg_layers_units"],
            post_cbhg_use_batch_norm=self.config["post_cbhg_use_batch_norm"],
        )

        return model

    def get_seq2seq(self):
        encoder = Seq2SeqEncoder(
            embedding_dim=self.config["encoder_embedding_dim"],
            inp_vocab_size=self.config["len_input_symbols"],
            layers_units=self.config["encoder_units"],
            use_batch_norm=self.config['use_batch_norm']
        )

        decoder = TacotronDecoder(
            self.config["len_target_symbols"],
            start_symbol_id=self.text_encoder.start_symbol_id,
            embedding_dim=self.config["decoder_embedding_dim"],
            encoder_dim=self.config["encoder_dim"],
            decoder_units=self.config["decoder_units"],
            decoder_layers=self.config["decoder_layers"],
            attention_type=self.config["attention_type"],
            attention_units=self.config["attention_units"],
            is_attention_accumulative=self.config["is_attention_accumulative"],
            use_prenet=self.config["use_decoder_prenet"],
            prenet_depth=self.config["decoder_prenet_depth"],
            teacher_forcing_probability=self.config["teacher_forcing_probability"],
        )

        model = Tacotron(encoder=encoder, decoder=decoder)

        return model

    def get_tacotron_based(self):
        encoder = TacotronEncoder(
            embedding_dim=self.config["encoder_embedding_dim"],
            inp_vocab_size=self.config["len_input_symbols"],
            prenet_sizes=self.config["prenet_sizes"],
            use_prenet=self.config["use_encoder_prenet"],
            cbhg_gru_units=self.config["cbhg_gru_units"],
            cbhg_filters=self.config["cbhg_filters"],
            cbhg_projections=self.config["cbhg_projections"],
        )

        decoder = TacotronDecoder(
            self.config["len_target_symbols"],
            start_symbol_id=self.text_encoder.start_symbol_id,
            embedding_dim=self.config["decoder_embedding_dim"],
            encoder_dim=self.config["encoder_dim"],
            decoder_units=self.config["decoder_units"],
            decoder_layers=self.config["decoder_layers"],
            attention_type=self.config["attention_type"],
            attention_units=self.config["attention_units"],
            is_attention_accumulative=self.config["is_attention_accumulative"],
            use_prenet=self.config["use_decoder_prenet"],
            prenet_depth=self.config["decoder_prenet_depth"],
            teacher_forcing_probability=self.config["teacher_forcing_probability"],
        )

        model = Tacotron(encoder=encoder, decoder=decoder)

        return model

    def get_text_encoder(self):
        """Getting the class of TextEncoder from config"""
        if self.config["text_cleaner"] not in [
            "basic_cleaners",
            "valid_arabic_cleaners",
            None,
        ]:
            raise Exception(f"cleaner is not known {self.config['text_cleaner']}")

        if self.config["text_encoder"] == "BasicArabicEncoder":
            text_encoder = BasicArabicEncoder(cleaner_fn=self.config["text_cleaner"])
        elif self.config["text_encoder"] == "ArabicEncoderWithStartSymbol":
            text_encoder = ArabicEncoderWithStartSymbol(
                cleaner_fn=self.config["text_cleaner"]
            )
        else:
            raise Exception(
                f"the text encoder is not found {self.config['text_encoder']}"
            )

        return text_encoder

    def get_loss_type(self):
        try:
            loss_type = LossType[self.config["loss_type"]]
        except:
            raise Exception(f"The loss type is not correct {self.config['loss_type']}")
        return loss_type


if __name__ == "__main__":
    config_path = "config/tacotron-base-config.yml"
    model_kind = "tacotron"
    config = ConfigManager(config_path=config_path, model_kind=model_kind)
