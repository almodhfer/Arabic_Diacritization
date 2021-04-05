import os
from typing import Dict

from diacritization_evaluation import der, wer
import torch
from torch import nn
from torch import optim
from torch.cuda.amp import autocast
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from tqdm import trange

from config_manager import ConfigManager
from dataset import load_iterators
from diacritizer import CBHGDiacritizer, Seq2SeqDiacritizer
from util.learning_rates import LearningRateDecay
from options import OptimizerType
from util.utils import (
    categorical_accuracy,
    count_parameters,
    initialize_weights,
    plot_alignment,
    repeater,
)


class Trainer:
    def run(self):
        raise NotImplementedError


class GeneralTrainer(Trainer):
    def __init__(self, config_path: str, model_kind: str) -> None:
        self.config_path = config_path
        self.model_kind = model_kind
        self.config_manager = ConfigManager(
            config_path=config_path, model_kind=model_kind
        )
        self.config = self.config_manager.config
        self.losses = []
        self.lr = 0
        self.pad_idx = 0
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.set_device()

        self.config_manager.create_remove_dirs()
        self.text_encoder = self.config_manager.text_encoder
        self.start_symbol_id = self.text_encoder.start_symbol_id
        self.summary_manager = SummaryWriter(log_dir=self.config_manager.log_dir)

        self.model = self.config_manager.get_model()

        self.optimizer = self.get_optimizer()
        self.model = self.model.to(self.device)

        self.load_model(model_path=self.config.get("train_resume_model_path"))
        self.load_diacritizer()

        self.initialize_model()

        self.print_config()

    def set_device(self):
        if self.config.get("device"):
            self.device = self.config["device"]
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def print_config(self):
        self.config_manager.dump_config()
        self.config_manager.print_config()

        if self.global_step > 1:
            print(f"loaded form {self.global_step}")

        parameters_count = count_parameters(self.model)
        print(f"The model has {parameters_count} trainable parameters parameters")

    def load_diacritizer(self):
        if self.model_kind in ["cbhg", "baseline"]:
            self.diacritizer = CBHGDiacritizer(self.config_path, self.model_kind)
        elif self.model_kind in ["seq2seq", "tacotron_based"]:
            self.diacritizer = Seq2SeqDiacritizer(self.config_path, self.model_kind)

    def initialize_model(self):
        if self.global_step > 1:
            return
        if self.model_kind == "transformer":
            print("Initializing using xavier_uniform_")
            self.model.apply(initialize_weights)

    def print_losses(self, step_results, tqdm):
        self.summary_manager.add_scalar(
            "loss/loss", step_results["loss"], global_step=self.global_step
        )

        tqdm.display(f"loss: {step_results['loss']}", pos=3)
        for pos, n_steps in enumerate(self.config["n_steps_avg_losses"]):
            if len(self.losses) > n_steps:

                self.summary_manager.add_scalar(
                    f"loss/loss-{n_steps}",
                    sum(self.losses[-n_steps:]) / n_steps,
                    global_step=self.global_step,
                )
                tqdm.display(
                    f"{n_steps}-steps average loss: {sum(self.losses[-n_steps:]) / n_steps}",
                    pos=pos + 4,
                )

    def evaluate(self, iterator, tqdm, use_target=True):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        tqdm.set_description(f"Eval: {self.global_step}")
        with torch.no_grad():
            for batch_inputs in iterator:
                batch_inputs["src"] = batch_inputs["src"].to(self.device)
                batch_inputs["lengths"] = batch_inputs["lengths"].to("cpu")
                if use_target:
                    batch_inputs["target"] = batch_inputs["target"].to(self.device)
                else:
                    batch_inputs["target"] = None

                outputs = self.model(
                    src=batch_inputs["src"],
                    target=batch_inputs["target"],
                    lengths=batch_inputs["lengths"],
                )

                predictions = outputs["diacritics"]

                predictions = predictions.view(-1, predictions.shape[-1])
                targets = batch_inputs["target"]
                targets = targets.view(-1)
                loss = self.criterion(predictions, targets.to(self.device))
                acc = categorical_accuracy(
                    predictions, targets.to(self.device), self.pad_idx
                )
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                tqdm.update()

        tqdm.reset()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate_with_error_rates(self, iterator, tqdm):
        all_orig = []
        all_predicted = []
        results = {}
        self.diacritizer.set_model(self.model)
        evaluated_batches = 0
        tqdm.set_description(f"Calculating DER/WER {self.global_step}: ")
        for batch in iterator:
            if evaluated_batches > int(self.config["error_rates_n_batches"]):
                break

            predicted = self.diacritizer.diacritize_batch(batch)
            all_predicted += predicted
            all_orig += batch["original"]
            tqdm.update()

        summary_texts = []
        orig_path = os.path.join(self.config_manager.prediction_dir, f"original.txt")
        predicted_path = os.path.join(
            self.config_manager.prediction_dir, f"predicted.txt"
        )

        with open(orig_path, "w", encoding="utf8") as file:
            for sentence in all_orig:
                file.write(f"{sentence}\n")

        with open(predicted_path, "w", encoding="utf8") as file:
            for sentence in all_predicted:
                file.write(f"{sentence}\n")

        for i in range(int(self.config["n_predicted_text_tensorboard"])):
            if i > len(all_predicted):
                break

            summary_texts.append(
                (f"eval-text/{i}", f"{ all_orig[i]} |->  {all_predicted[i]}")
            )

        results["DER"] = der.calculate_der_from_path(orig_path, predicted_path)
        results["DER*"] = der.calculate_der_from_path(
            orig_path, predicted_path, case_ending=False
        )
        results["WER"] = wer.calculate_wer_from_path(orig_path, predicted_path)
        results["WER*"] = wer.calculate_wer_from_path(
            orig_path, predicted_path, case_ending=False
        )
        tqdm.reset()
        return results, summary_texts

    def run(self):
        scaler = torch.cuda.amp.GradScaler()
        train_iterator, _, validation_iterator = load_iterators(self.config_manager)
        print("data loaded")
        print("----------------------------------------------------------")
        tqdm_eval = trange(0, len(validation_iterator), leave=True)
        tqdm_error_rates = trange(0, len(validation_iterator), leave=True)
        tqdm_eval.set_description("Eval")
        tqdm_error_rates.set_description("WER/DER : ")
        tqdm = trange(self.global_step, self.config["max_steps"] + 1, leave=True)

        for batch_inputs in repeater(train_iterator):
            tqdm.set_description(f"Global Step {self.global_step}")
            if self.config["use_decay"]:
                self.lr = self.adjust_learning_rate(
                    self.optimizer, global_step=self.global_step
                )
            self.optimizer.zero_grad()
            if self.device == "cuda" and self.config["use_mixed_precision"]:
                with autocast():
                    step_results = self.run_one_step(batch_inputs)
                    scaler.scale(step_results["loss"]).backward()
                    scaler.unscale_(self.optimizer)
                    if self.config.get("CLIP"):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config["CLIP"]
                        )

                    scaler.step(self.optimizer)

                    scaler.update()
            else:
                step_results = self.run_one_step(batch_inputs)

                loss = step_results["loss"]
                loss.backward()
                if self.config.get("CLIP"):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["CLIP"]
                    )
                self.optimizer.step()

            self.losses.append(step_results["loss"].item())

            self.print_losses(step_results, tqdm)

            self.summary_manager.add_scalar(
                "meta/learning_rate", self.lr, global_step=self.global_step
            )

            if self.global_step % self.config["model_save_frequency"] == 0:
                torch.save(
                    {
                        "global_step": self.global_step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    os.path.join(
                        self.config_manager.models_dir,
                        f"{self.global_step}-snapshot.pt",
                    ),
                )

            if self.global_step % self.config["evaluate_frequency"] == 0:
                loss, acc = self.evaluate(validation_iterator, tqdm_eval)
                self.summary_manager.add_scalar(
                    "evaluate/loss", loss, global_step=self.global_step
                )
                self.summary_manager.add_scalar(
                    "evaluate/acc", acc, global_step=self.global_step
                )
                tqdm.display(
                    f"Evaluate {self.global_step}: accuracy, {acc}, loss: {loss}", pos=8
                )
                self.model.train()

            if (
                self.global_step % self.config["evaluate_with_error_rates_frequency"]
                == 0
            ):
                error_rates, summery_texts = self.evaluate_with_error_rates(
                    validation_iterator, tqdm_error_rates
                )
                if error_rates:
                    WER = error_rates["WER"]
                    DER = error_rates["DER"]
                    DER1 = error_rates["DER*"]
                    WER1 = error_rates["WER*"]

                    self.summary_manager.add_scalar(
                        "error_rates/WER",
                        WER / 100,
                        global_step=self.global_step,
                    )
                    self.summary_manager.add_scalar(
                        "error_rates/DER",
                        DER / 100,
                        global_step=self.global_step,
                    )
                    self.summary_manager.add_scalar(
                        "error_rates/DER*",
                        DER1 / 100,
                        global_step=self.global_step,
                    )
                    self.summary_manager.add_scalar(
                        "error_rates/WER*",
                        WER1 / 100,
                        global_step=self.global_step,
                    )

                    error_rates = f"DER: {DER}, WER: {WER}, DER*: {DER1}, WER*: {WER1}"
                    tqdm.display(f"WER/DER {self.global_step}: {error_rates}", pos=9)

                    for tag, text in summery_texts:
                        self.summary_manager.add_text(tag, text)

                self.model.train()

            if self.global_step % self.config["train_plotting_frequency"] == 0:
                self.plot_attention(step_results)

            self.report(step_results, tqdm)

            self.global_step += 1
            if self.global_step > self.config["max_steps"]:
                print("Training Done.")
                return

            tqdm.update()

    def run_one_step(self, batch_inputs: Dict[str, torch.Tensor]):
        batch_inputs["src"] = batch_inputs["src"].to(self.device)
        batch_inputs["lengths"] = batch_inputs["lengths"].to("cpu")
        batch_inputs["target"] = batch_inputs["target"].to(self.device)

        outputs = self.model(
            src=batch_inputs["src"],
            target=batch_inputs["target"],
            lengths=batch_inputs["lengths"],
        )

        predictions = outputs["diacritics"].contiguous()
        targets = batch_inputs["target"].contiguous()
        predictions = predictions.view(-1, predictions.shape[-1])
        targets = targets.view(-1)
        loss = self.criterion(predictions.to(self.device), targets.to(self.device))
        outputs.update({"loss": loss})
        return outputs

    def predict(self, iterator):
        pass

    def load_model(self, model_path: str = None, load_optimizer: bool = True):
        with open(
            self.config_manager.base_dir / f"{self.model_kind}_network.txt", "w"
        ) as file:
            file.write(str(self.model))

        if model_path is None:
            last_model_path = self.config_manager.get_last_model_path()
            if last_model_path is None:
                self.global_step = 1
                return
        else:
            last_model_path = model_path

        print(f"loading from {last_model_path}")
        saved_model = torch.load(last_model_path)
        self.model.load_state_dict(saved_model["model_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(saved_model["optimizer_state_dict"])
        self.global_step = saved_model["global_step"] + 1

    def get_optimizer(self):
        if self.config["optimizer"] == OptimizerType.Adam:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == OptimizerType.SGD:
            optimizer = optim.SGD(
                self.model.parameters(), lr=self.config["learning_rate"], momentum=0.9
            )
        else:
            raise ValueError("Optimizer option is not valid")

        return optimizer

    def get_learning_rate(self):
        return LearningRateDecay(
            lr=self.config["learning_rate"],
            warmup_steps=self.config.get("warmup_steps", 4000.0),
        )

    def adjust_learning_rate(self, optimizer, global_step):
        learning_rate = self.get_learning_rate()(global_step=global_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
        return learning_rate

    def plot_attention(self, results):
        pass

    def report(self, results, tqdm):
        pass


class Seq2SeqTrainer(GeneralTrainer):
    def plot_attention(self, results):
        plot_alignment(
            results["attention"][0],
            str(self.config_manager.plot_dir),
            self.global_step,
        )

        self.summary_manager.add_image(
            "Train/attention",
            results["attention"][0].unsqueeze(0),
            global_step=self.global_step,
        )


class CBHGTrainer(GeneralTrainer):
    pass
