"""
Types of various choices used during training
"""
from enum import Enum


class AttentionType(Enum):
    """Type of attention used during training"""

    LocationSensitive = 1
    Content_Based = 2
    MultiHead = 3


class LearningRateType(Enum):
    """Type of learning rate used during training"""

    Learning_Rate_Decay = 1
    Cosine_Scheduler = 2
    SquareRoot_Scheduler = 3


class OptimizerType(Enum):
    """Type of optimizer used during training"""

    Adam = 1
    SGD = 2
    AdamW = 3


class LossType(Enum):
    """Type of loss function used during training"""

    L1_LOSS = 1
    MSE_LOSS = 2
    L1_LOSS_MASKED = 3
    MSE_LOSS_MASKED = 4
    BOTH = 5
    BOTH_MASKED = 6
