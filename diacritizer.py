from typing import Dict
import torch
from config_manager import ConfigManager


class Diacritizer:
    def __init__(
        self, config_path: str, model_kind: str, load_model: bool = False
    ) -> None:
        self.config_path = config_path
        self.model_kind = model_kind
        self.config_manager = ConfigManager(
            config_path=config_path, model_kind=model_kind
        )
        self.config = self.config_manager.config
        self.text_encoder = self.config_manager.text_encoder
        if self.config.get("device"):
            self.device = self.config["device"]
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if load_model:
            self.model, self.global_step = self.config_manager.load_model()
            self.model = self.model.to(self.device)

        self.start_symbol_id = self.text_encoder.start_symbol_id

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def diacritize_text(self, text: str):
        seq = self.text_encoder.input_to_sequence(text)
        output = self.diacritize_batch(torch.LongTensor([seq]).to(self.device))

    def diacritize_batch(self, batch):
        raise NotImplementedError()

    def diacritize_iterators(self, iterator):
        pass


class CBHGDiacritizer(Diacritizer):
    def diacritize_batch(self, batch):
        self.model.eval()
        inputs = batch["src"]
        lengths = batch["lengths"]
        outputs = self.model(inputs.to(self.device), lengths.to("cpu"))
        diacritics = outputs["diacritics"]
        predictions = torch.max(diacritics, 2).indices
        sentences = []

        for src, prediction in zip(inputs, predictions):
            sentence = self.text_encoder.combine_text_and_haraqat(
                list(src.detach().cpu().numpy()),
                list(prediction.detach().cpu().numpy()),
            )
            sentences.append(sentence)

        return sentences


class Seq2SeqDiacritizer(Diacritizer):
    def diacritize_batch(self, batch):
        self.model.eval()
        inputs = batch["src"]
        lengths = batch["lengths"]
        outputs = self.model(inputs.to(self.device), lengths.to("cpu"))
        diacritics = outputs["diacritics"]
        predictions = torch.max(diacritics, 2).indices
        sentences = []

        for src, prediction in zip(inputs, predictions):
            sentence = self.text_encoder.combine_text_and_haraqat(
                list(src.detach().cpu().numpy()),
                list(prediction.detach().cpu().numpy()),
            )
            sentences.append(sentence)

        return sentences
