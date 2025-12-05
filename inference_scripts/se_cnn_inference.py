import os
import time
import cv2
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

MODELS = [
    ("SE_Reduction_4",  4,  "/models/customcnn_depth4_se_r4/customcnn_depth4_se_r4_final.pth"),
    ("SE_Reduction_8",  8,  "/models/customcnn_depth4_se_r8/customcnn_depth4_se_r8_final.pth"),
    ("SE_Reduction_16", 16, "/models/customcnn_depth4_se_r16/customcnn_depth4_se_r16_final.pth"),
]

DATASET_ROOT = "/content/dataset_cropped_224x224_newest/dataset_cropped_224x224_newest"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR   = os.path.join(DATASET_ROOT, "valid")

class MicroplasticsDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_dir):
        self.df = df
        self.images_dir = images_dir
        self.image_files = df["filename"].unique().tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        records = self.df[self.df["filename"] == img_file]
        boxes = records[["xmin", "ymin", "xmax", "ymax"]].values
        labels = records["label"].values

        # Normalize to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return img, {"boxes": boxes, "labels": labels}


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

if not os.path.exists(os.path.join(VAL_DIR, "_annotations.csv")):
    print(f"Warning: Annotation file not found at {VAL_DIR}")
    train_df = None
    val_loader = None
else:
    train_df = pd.read_csv(os.path.join(TRAIN_DIR, "_annotations.csv"))
    val_df   = pd.read_csv(os.path.join(VAL_DIR, "_annotations.csv"))

    unique_classes = sorted(train_df["class"].unique())
    class_to_idx = {cls: i + 1 for i, cls in enumerate(unique_classes)}

    train_df["label"] = train_df["class"].map(class_to_idx)
    val_df["label"]   = val_df["class"].map(class_to_idx)

    val_dataset = MicroplasticsDataset(val_df, VAL_DIR)
    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=1)

    print("Validation images:", len(val_dataset))
    print("Classes:", unique_classes)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CustomCNNBackboneSE(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), se_reduction=16):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools  = nn.ModuleList()

        in_ch = 3
        for ch in channels:
            block = nn.Sequential(
                nn.Conv2d(in_ch, ch, 3, 1, 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.Conv2d(ch, ch, 3, 1, 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                SEBlock(channels=ch, reduction=se_reduction),
            )
            self.blocks.append(block)
            self.pools.append(nn.MaxPool2d(2))
            in_ch = ch

        self.out_channels = channels[-1]

    def forward(self, x):
        for block, pool in zip(self.blocks, self.pools):
            x = pool(block(x))
        return x

def build_model(se_reduction, num_classes):
    # Fixed channels for "depth4" models based on your training script
    channels = (32, 64, 128, 256)
    
    backbone = CustomCNNBackboneSE(channels=channels, se_reduction=se_reduction)
    
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )

    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=224,
        max_size=224,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
    ).to(device)


try:
    from thop import profile
    thop_available = True
except Exception:
    thop_available = False
    print("THOP not available. MACs = N/A")


def compute_params(model):
    return sum(p.numel() for p in model.backbone.parameters()) / 1e6


def compute_macs(model):
    if not thop_available:
        return None
    dummy = torch.randn(1, 3, 224, 224).to(device)
    macs, _ = profile(model.backbone, inputs=(dummy,), verbose=False)
    return macs / 1e9


def evaluate_map(model):
    model.eval()
    metric = MeanAveragePrecision().to(device)

    with torch.no_grad():
        for images, targets in val_loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds = model(images)
            metric.update(preds, targets)

    m = metric.compute()
    return float(m["map_50"]), float(m["map"])


def measure_fps(model):
    model.eval()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    n = 0
    with torch.no_grad():
        for images, _ in val_loader:
            images = [i.to(device) for i in images]
            _ = model(images)
            n += len(images)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return n / (time.time() - t0)


def print_table(rows):
    headers = ["Model", "Reduction", "mAP@50", "mAP@50:95", "FPS", "Bkbn Params(M)", "Bkbn MACs(G)"]
    w = [len(h) for h in headers]

    for r in rows:
        for i, v in enumerate(r):
            w[i] = max(w[i], len(str(v)))

    def sep(c="-"):
        return "+-" + "-+-".join(c * wi for wi in w) + "-+"

    def row(vals):
        return "| " + " | ".join(str(vals[i]).ljust(w[i]) for i in range(len(vals))) + " |"

    print()
    print(sep("-"))
    print(row(headers))
    print(sep("="))
    for r in rows:
        print(row(r))
        print(sep("-"))

def main():
    if val_loader is None:
        return

    rows = []
    num_classes = len(unique_classes) + 1

    for tag, reduction, ckpt in MODELS:
        if not os.path.isfile(ckpt):
            print(f"Missing checkpoint file: {ckpt}")
            continue

        print(f"Evaluating: {tag} (Reduction={reduction})...")

        model = build_model(reduction, num_classes)
        
        try:
            state_dict = torch.load(ckpt, map_location=device)

            clean_state_dict = {
                k: v for k, v in state_dict.items() 
                if "total_ops" not in k and "total_params" not in k
            }
            
            model.load_state_dict(clean_state_dict)
            
        except Exception as e:
            print(f"Error loading weights for {tag}: {e}")
            continue

        params_M = compute_params(model)
        macs_G = compute_macs(model)
        map50, map5095 = evaluate_map(model)
        fps = measure_fps(model)

        rows.append([
            tag,
            str(reduction),
            f"{map50:.3f}",
            f"{map5095:.3f}",
            f"{fps:.2f}",
            f"{params_M:.2f}",
            "N/A" if macs_G is None else f"{macs_G:.2f}",
        ])

    if rows:
        print_table(rows)
    else:
        print("No models evaluated.")

if __name__ == "__main__":
    main()