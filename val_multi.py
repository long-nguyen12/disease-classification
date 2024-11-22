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

from datasets import *
from datasets.transforms import get_train_transforms, get_val_transforms
from models import *
from utils.losses import CrossEntropyLoss, LabelSmoothCrossEntropy
from utils.metrics import compute_accuracy
from utils.utils import create_progress_bar, fix_seeds, setup_cudnn

console = Console()


def test(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, top1_acc, top5_acc = 0, 0, 0

    top1_acc_1 = 0
    top1_acc_2 = 0
    top1_acc_3 = 0
    top1_acc_4 = 0

    top5_acc_1 = 0
    top5_acc_2 = 0
    top5_acc_3 = 0
    top5_acc_4 = 0
    infer_time = 0
    with torch.no_grad():
        for X, y in dataloader:
            start = time.time()
            X, y = X.to(device), y.to(device)
            preds = model(X)
            end = time.time() - start
            infer_time += end
            for i, pred in enumerate(preds):
                test_loss += loss_fn(pred, y).item()
                acc1, acc5 = compute_accuracy(pred, y, topk=(1, 4))

                if i + 1 == 1:
                    top1_acc_1 += acc1 * X.shape[0]
                    top5_acc_1 += acc5 * X.shape[0]
                elif i + 1 == 2:
                    top1_acc_2 += acc1 * X.shape[0]
                    top5_acc_2 += acc5 * X.shape[0]
                elif i + 1 == 3:
                    top1_acc_3 += acc1 * X.shape[0]
                    top5_acc_3 += acc5 * X.shape[0]
                else:
                    top1_acc_4 += acc1 * X.shape[0]
                    top5_acc_4 += acc5 * X.shape[0]

    test_loss /= num_batches

    top1_acc_1 /= size
    top1_acc_2 /= size
    top1_acc_3 /= size
    top1_acc_4 /= size

    top5_acc_1 /= size
    top5_acc_2 /= size
    top5_acc_3 /= size
    top5_acc_4 /= size

    infer_time /= size

    console.print(
        f"\n Average inference time: [blue]{(infer_time):>0.5f}[/blue],\n Top-1 Exit-1 Accuracy: [blue]{(top1_acc_1):>0.1f}%[/blue],\n Top-1 Exit-2 Accuracy: [blue]{(top1_acc_2):>0.1f}%[/blue],\n Top-1 Exit-3 Accuracy: [blue]{(top1_acc_3):>0.1f}%[/blue],\n Top-1 Exit-4 Accuracy: [blue]{(top1_acc_4):>0.1f}%[/blue],\tAvg Loss: [blue]{test_loss:>8f}[/blue]"
    )
    return top1_acc_4, top5_acc_4


def main(cfg: argparse.Namespace):
    start = time.time()
    save_dir = Path(cfg.SAVE_DIR)
    save_dir.mkdir(exist_ok=True)
    fix_seeds(42)
    setup_cudnn()

    device = torch.device(cfg.DEVICE)
    num_workers = 8

    # dataloader
    
    if cfg.DATASET_NAME == "Kvasir":
        DiseaseDataset = KvasirDataLoader(
            cfg.DATASET, 1, cfg.IMAGE_SIZE, num_workers
        )
    elif cfg.DATASET_NAME == "Ham10000":
        DiseaseDataset = HamDataloader(
            "data/ham10000/ham10000-train.csv",
            "data/ham10000/ham10000-test.csv",
            1,
            cfg.IMAGE_SIZE,
            None,
            num_workers,
        )
    else:
        DiseaseDataset = DiseaseDataloader(
            cfg.DATASET, 1, cfg.IMAGE_SIZE, num_workers
        )
        
    trainloader, testloader = DiseaseDataset.get_data_loaders()
    # initialize model and load imagenet pretrained
    model = eval(cfg.MODEL)(cfg.VARIANT, cfg.PRETRAINED, cfg.CLASSES, cfg.IMAGE_SIZE)
    model = model.to(device)
    
    state_dict_path = f"output/{cfg.DATASET_NAME}_{cfg.MODEL}_50_last.pth"
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    val_loss_fn = CrossEntropyLoss()

    top1_acc, top5_acc = test(testloader, model, val_loss_fn, device)

    end = time.gmtime(time.time() - start)

    table = Table(show_header=True, header_style="magenta")
    table.add_column("Best Top-1 Accuracy")
    table.add_column("Best Top-5 Accuracy")
    table.add_column("Total Training Time")
    table.add_row(f"{top1_acc}%", f"{top5_acc}%", f"{end}")
    console.print(table)


if __name__ == "__main__":
    configs = ["ham10000.yaml", "ear.yaml", "kvasir.yaml"]
    for config in configs:
        config_path = f"configs/{config}"
        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg", type=str, default=config_path)
        args = parser.parse_args()
        cfg = argparse.Namespace(**yaml.load(open(args.cfg), Loader=yaml.SafeLoader))
        main(cfg)
