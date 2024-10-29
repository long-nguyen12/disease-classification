import argparse
import time
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.table import Table
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import *

from datasets.disease import DiseaseDataloader
from datasets.transforms import get_train_transforms, get_val_transforms
from models import *
from utils.losses import CrossEntropyLoss, LabelSmoothCrossEntropy
from utils.metrics import compute_accuracy
from utils.utils import create_progress_bar, fix_seeds, setup_cudnn

from thop import profile, clever_format

def CalParams(model, x):
    flops, params = profile(model, inputs=(x, ))
    flops, params = clever_format([flops, params], "%.3f")
    print("\nFLOP: {}\nParams: {}".format(flops, params))

def main(cfg: argparse.Namespace):
    device = torch.device(cfg.DEVICE)
    # initialize model and load imagenet pretrained
    model = eval(cfg.MODEL)(cfg.VARIANT, cfg.PRETRAINED, cfg.CLASSES, cfg.IMAGE_SIZE)

    model = model.to(device)

    x = torch.randn(1, 3, 224, 224).to(device)
    CalParams(model, x)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/finetune.yaml")
    args = parser.parse_args()
    cfg = argparse.Namespace(**yaml.load(open(args.cfg), Loader=yaml.SafeLoader))
    main(cfg)
