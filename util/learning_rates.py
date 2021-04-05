import numpy as np
import math


class LearningRateDecay:
    def __init__(self, lr=0.002, warmup_steps=4000.0) -> None:
        self.lr = lr
        self.warmup_steps = warmup_steps

    def __call__(self, global_step) -> float:
        step = global_step + 1.0
        lr = (
            self.lr
            * self.warmup_steps ** 0.5
            * np.minimum(step * self.warmup_steps ** -1.5, step ** -0.5)
        )

        return lr

class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, global_step):
        global_step = global_step // 1000
        return self.lr * pow(global_step + 1.0, -0.5)


class CosineScheduler:
    def __init__(
        self, max_update, base_lr=0.02, final_lr=0, warmup_steps=0, warmup_begin_lr=0
    ):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, global_step):
        increase = (
            (self.base_lr_orig - self.warmup_begin_lr)
            * float(global_step)
            / float(self.warmup_steps)
        )
        return self.warmup_begin_lr + increase

    def __call__(self, global_step):
        if global_step < self.warmup_steps:
            return self.get_warmup_lr(global_step)
        if global_step <= self.max_update:
            self.base_lr = (
                self.final_lr
                + (self.base_lr_orig - self.final_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (global_step - self.warmup_steps) / self.max_steps
                    )
                )
                / 2
            )
        return self.base_lr

def adjust_learning_rate(optimizer, global_step):
    lr = LearningRateDecay()(global_step=global_step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

