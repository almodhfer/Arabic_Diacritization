import argparse
import random
from tester import DiacritizationTester

import numpy as np
import torch


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_kind", type=str, required=True)
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--model_path", dest="model_path", type=str, required=False)
    return parser


parser = train_parser()
args = parser.parse_args()

tester = DiacritizationTester(args.config, args.model_kind)
tester.run()
