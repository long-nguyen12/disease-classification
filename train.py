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


def train(dataloader, model, loss_fn, optimizer, scheduler, scaler, device, epoch, cfg):
    model.train()
    progress = create_progress_bar()
    lr = scheduler.get_last_lr()[0]
    task_id = progress.add_task(
        description="",
        total=len(dataloader),
        epoch=epoch + 1,
        epochs=cfg.EPOCHS,
        lr=lr,
        loss=0.0,
    )

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        loss = 0.0
        with autocast(enabled=cfg.AMP):
            preds = model(X)
            for pred in preds:
                loss += loss_fn(pred, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        progress.update(
            task_id, description="", advance=1, refresh=True, loss=loss.item()
        )
    scheduler.step()
    progress.stop()


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

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
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

    console.print(
        f"\n Top-1 Exit-1 Accuracy: [blue]{(top1_acc_1):>0.1f}%[/blue],\n Top-1 Exit-2 Accuracy: [blue]{(top1_acc_2):>0.1f}%[/blue],\n Top-1 Exit-3 Accuracy: [blue]{(top1_acc_3):>0.1f}%[/blue],\n Top-1 Exit-4 Accuracy: [blue]{(top1_acc_4):>0.1f}%[/blue],\tAvg Loss: [blue]{test_loss:>8f}[/blue]"
    )
    return top1_acc_4, top5_acc_4


def main(cfg: argparse.Namespace):
    start = time.time()
    save_dir = Path(cfg.SAVE_DIR)
    save_dir.mkdir(exist_ok=True)
    fix_seeds(42)
    setup_cudnn()
    best_top1_acc, best_top5_acc = 0.0, 0.0

    device = torch.device(cfg.DEVICE)
    num_workers = 8

    # dataloader
    # DiseaseDataset = DiseaseDataloader(
    #     cfg.DATASET, cfg.BATCH_SIZE, cfg.IMAGE_SIZE, num_workers
    # )
    DiseaseDataset = HamDataloader(
        'data/ham10000/ham10000-train.csv', 'data/ham10000/ham10000-test.csv', cfg.BATCH_SIZE, cfg.IMAGE_SIZE, None, num_workers
    )
    trainloader, testloader = DiseaseDataset.get_data_loaders()
    # initialize model and load imagenet pretrained
    model = eval(cfg.MODEL)(cfg.VARIANT, cfg.PRETRAINED, cfg.CLASSES, cfg.IMAGE_SIZE)

    # freeze layers or not
    if cfg.FREEZE:
        for n, p in model.named_parameters():
            if "head" not in n:
                p.requires_grad_ = False

    model = model.to(device)
    train_loss_fn = LabelSmoothCrossEntropy(smoothing=0.1)
    val_loss_fn = CrossEntropyLoss()
    # optimizer = AdamW(
    #     model.parameters(), cfg.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    # )

    optimizer = Adam(model.parameters(), cfg.LR)
    # scheduler = StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(trainloader) * cfg.EPOCHS,
        eta_min=cfg.LR / 1000,
    )
    scaler = GradScaler(enabled=cfg.AMP)

    for epoch in range(cfg.EPOCHS):
        train(
            trainloader,
            model,
            train_loss_fn,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            cfg,
        )

        if (epoch + 1) % cfg.EVAL_INTERVAL == 0 or (epoch + 1) == cfg.EPOCHS:
            top1_acc, top5_acc = test(testloader, model, val_loss_fn, device)

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_top5_acc = top5_acc
                torch.save(
                    model.state_dict(),
                    save_dir / f"{cfg.DATASET_NAME}_{cfg.MODEL}_{cfg.VARIANT}_last.pth",
                )
            console.print(
                f" Best Top-1 Accuracy: [red]{(best_top1_acc):>0.1f}%[/red]\tBest Top-5 Accuracy: [red]{(best_top5_acc):>0.1f}%[/red]\n"
            )

    end = time.gmtime(time.time() - start)
    total_time = time.strftime("%H:%M:%S", end)

    table = Table(show_header=True, header_style="magenta")
    table.add_column("Best Top-1 Accuracy")
    table.add_column("Best Top-5 Accuracy")
    table.add_column("Total Training Time")
    table.add_row(f"{best_top1_acc}%", f"{best_top5_acc}%", str(total_time))
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/ham10000.yaml")
    args = parser.parse_args()
    cfg = argparse.Namespace(**yaml.load(open(args.cfg), Loader=yaml.SafeLoader))
    main(cfg)
