# IMPORTANT: Unzip dataset at ./content/dataset_cropped_224x224_newest before running!

import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import albumentations as A

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from datetime import timedelta

# Utility
def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Dataset
class MicroplasticsDataset(Dataset):
    def __init__(self, df, images_dir, transforms=None):
        self.df = df
        self.images_dir = images_dir
        self.transforms = transforms
        self.image_files = df['filename'].unique().tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        path = os.path.join(self.images_dir, img_file)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        samples = self.df[self.df['filename'] == img_file]
        boxes  = samples[['xmin','ymin','xmax','ymax']].values
        labels = samples['label'].values

        if self.transforms:
            aug = self.transforms(image=img, bboxes=boxes, class_labels=labels)
            img = aug["image"]
            boxes = aug["bboxes"]
            labels = aug["class_labels"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return img, {"boxes": boxes, "labels": labels}


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# Data preparation
crop_train_dir = "./content/dataset_cropped_224x224_newest/dataset_cropped_224x224_newest/train"
crop_val_dir   = "./content/dataset_cropped_224x224_newest/dataset_cropped_224x224_newest/valid"

train_df = pd.read_csv(os.path.join(crop_train_dir, "_annotations.csv"))
val_df   = pd.read_csv(os.path.join(crop_val_dir, "_annotations.csv"))

unique_classes = sorted(train_df['class'].unique())
class_to_idx   = {cls: i + 1 for i, cls in enumerate(unique_classes)}

train_df["label"] = train_df["class"].map(class_to_idx)
val_df["label"]   = val_df["class"].map(class_to_idx)

train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, p=0.5),
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
)

train_dataset = MicroplasticsDataset(train_df, crop_train_dir, transforms=train_transforms)
val_dataset   = MicroplasticsDataset(val_df,   crop_val_dir,   transforms=None)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

# Model
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class CustomCNNBackboneRes(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools  = nn.ModuleList()
        in_ch = 3
        for ch in channels:
            self.blocks.append(ResidualBlock(in_ch, ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = ch
        self.out_channels = channels[-1]

    def forward(self, x):
        for block, pool in zip(self.blocks, self.pools):
            x = pool(block(x))
        return x


def build_model_residual(channels=(32, 64, 128, 256)):
    backbone = CustomCNNBackboneRes(channels)
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    model = FasterRCNN(
        backbone,
        num_classes=len(unique_classes) + 1,
        rpn_anchor_generator=anchor_generator,
        min_size=224,
        max_size=224,
        image_mean=[0, 0, 0],
        image_std=[1, 1, 1],
    ).to(device)
    return model

# Metrics utilities
try:
    from thop import profile
    thop_available = True
except ImportError:
    thop_available = False
    print("THOP not available, MACs will be set to None.")


def compute_model_macs(model, device):
    if not thop_available:
        return None, None

    # FasterRCNN expects list of 3D tensors (C, H, W)
    dummy_img = torch.randn(3, 224, 224).to(device)
    try:
        macs, params = profile(model, inputs=([dummy_img],), verbose=False)
    except Exception:
        print("THOP failed on full model — computing backbone only.")
        dummy_batch = torch.randn(1, 3, 224, 224).to(device)
        macs, params = profile(model.backbone, inputs=(dummy_batch,), verbose=False)
    return macs, params


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def compute_model_size_mb(model):
    # float32 params: 4 bytes each
    return count_parameters(model) * 4 / (1024 ** 2)


def measure_inference_speed(model, device, repeats=50):
    model.eval()
    dummy = torch.randn(3, 224, 224).to(device)

    # warm-up
    for _ in range(10):
        _ = model([dummy])

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(repeats):
        _ = model([dummy])
    if device.type == "cuda":
        torch.cuda.synchronize()

    total = time.time() - start
    ms_per_image = (total / repeats) * 1000.0
    fps = 1.0 / (total / repeats)
    return ms_per_image, fps

# Training wrapper
def run_residual_experiment(
    channels=(32, 64, 128, 256),
    num_epochs=30,
    tag="customcnn_residual",
    log_every=100,
    lr=1e-3,
    weight_decay=0.0,
    step_size=10,
    gamma=0.1,
):
    model = build_model_residual(channels)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )
    metric = MeanAveragePrecision().to(device)

    history = []

    base_results_dir = "./models"
    os.makedirs(base_results_dir, exist_ok=True)
    experiment_dir = os.path.join(base_results_dir, tag)
    os.makedirs(experiment_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        t_start = time.time()
        num_batches = len(train_loader)

        for batch_i, (images, targets) in enumerate(train_loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            losses = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_i % log_every == 0 or batch_i == 1:
                elapsed = time.time() - t_start
                eta = (elapsed / batch_i) * (num_batches - batch_i)
                print(
                    f"[Train] Batch {batch_i}/{num_batches} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg: {running_loss / batch_i:.4f} | "
                    f"ETA: {format_time(eta)}"
                )
                if torch.cuda.is_available():
                    print(f"GPU mem: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")

        scheduler.step()
        epoch_loss = running_loss / num_batches

        # Validation
        model.eval()
        metric.reset()
        print("\nValidating...")
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                preds = model(images)
                metric.update(preds, targets)

        m = metric.compute()
        map50 = float(m["map_50"])
        map5095 = float(m["map"])

        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "mAP_50": map50,
                "mAP_50_95": map5095,
                "channels": str(channels),
                "lr": lr,
                "weight_decay": weight_decay,
                "step_size": step_size,
                "gamma": gamma,
            }
        )

        print(f"\nEpoch {epoch} COMPLETED in {format_time(time.time() - epoch_start)}")
        print(f"Train Loss: {epoch_loss:.4f} | mAP@50: {map50:.3f} | mAP@50-95: {map5095:.3f}")
        print("--------------------------------------------------------")

    # Save training history and model weights
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(experiment_dir, f"{tag}_history.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(experiment_dir, f"{tag}_final.pth"))

    # Final metrics
    print("\nFinal Model Evaluation")
    macs, _ = compute_model_macs(model, device)
    total_params = count_parameters(model)
    model_size = compute_model_size_mb(model)
    ms, fps = measure_inference_speed(model, device)

    if macs is not None:
        print(f"MACs (approx): {macs / 1e6:.2f} M")
    else:
        print("MACs: None (THOP not available)")
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Inference: {ms:.2f} ms/img ({fps:.2f} FPS)")

    # Save metrics to text
    with open(os.path.join(experiment_dir, "metrics.txt"), "w") as f:
        f.write(f"Channels: {channels}\n")
        f.write(f"Best mAP@50: {df_hist['mAP_50'].max():.4f}\n")
        f.write(f"Best mAP@50-95: {df_hist['mAP_50_95'].max():.4f}\n")
        if macs is not None:
            f.write(f"MACs: {macs}\n")
            f.write(f"MACs (M): {macs / 1e6:.3f}\n")
        else:
            f.write("MACs: None (THOP not available)\n")
        f.write(f"Parameters: {total_params}\n")
        f.write(f"Model size (MB): {model_size:.2f}\n")
        f.write(f"Inference time (ms): {ms:.4f}\n")
        f.write(f"FPS: {fps:.4f}\n")
        f.write(f"LR: {lr}\n")
        f.write(f"Weight decay: {weight_decay}\n")
        f.write(f"Step size: {step_size}\n")
        f.write(f"Gamma: {gamma}\n")

    summary = {
        "tag": tag,
        "channels": str(channels),
        "lr": lr,
        "weight_decay": weight_decay,
        "step_size": step_size,
        "gamma": gamma,
        "best_mAP_50": df_hist["mAP_50"].max(),
        "best_mAP_50_95": df_hist["mAP_50_95"].max(),
        "MACs": macs,
        "params": total_params,
        "model_size_MB": model_size,
        "inference_ms": ms,
        "FPS": fps,
    }

    return df_hist, summary

# Main: 3 hyperparameter configs
if __name__ == "__main__":
    hp_configs = [
        {
            "name": "hp_fast",
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "step_size": 10,
            "gamma": 0.1,
        },
        {
            "name": "hp_balanced",
            "lr": 5e-4,
            "weight_decay": 1e-5,
            "step_size": 8,
            "gamma": 0.3,
        },
        {
            "name": "hp_stable",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "step_size": 5,
            "gamma": 0.5,
        },
    ]

    channels = (32, 64, 128, 256)
    all_summaries = []

    for cfg in hp_configs:
        tag = f"customcnn_depth4_residual_{cfg['name']}"
        print(f"\nRunning experiment: {tag}")
        print(
            f"Channels={channels}, "
            f"LR={cfg['lr']}, wd={cfg['weight_decay']}, "
            f"step_size={cfg['step_size']}, gamma={cfg['gamma']}"
        )

        df_hist_res, summary_res = run_residual_experiment(
            channels=channels,
            num_epochs=30,
            tag=tag,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            step_size=cfg["step_size"],
            gamma=cfg["gamma"],
        )
        all_summaries.append(summary_res)

    df_all = pd.DataFrame(all_summaries)
    base_results_dir = "./models"
    os.makedirs(base_results_dir, exist_ok=True)
    df_all.to_csv(os.path.join(base_results_dir, "hp_sweep_summary.csv"), index=False)

    print("\nHyperparameter sweep summary")
    print(df_all)
