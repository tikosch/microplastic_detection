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
    ("Residual_HP_Balanced", (32, 64, 128, 256), "/models/customcnn_depth4_residual_hp_balanced/customcnn_depth4_residual_hp_balanced_final.pth"),
    ("Residual_HP_Fast",     (32, 64, 128, 256), "/models/customcnn_depth4_residual_hp_fast/customcnn_depth4_residual_hp_fast_final.pth"),
    ("Residual_HP_Stable",   (32, 64, 128, 256), "/models/customcnn_depth4_residual_hp_stable/customcnn_depth4_residual_hp_stable_final.pth"),
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

# --- LOAD DATA ---
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


def build_model(channels, num_classes):
    backbone = CustomCNNBackboneRes(channels)
    
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
        image_mean=[0, 0, 0],
        image_std=[1, 1, 1],
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
    headers = ["Model", "Channels", "mAP@50", "mAP@50:95", "FPS", "Bkbn Params(M)", "Bkbn MACs(G)"]
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

    for tag, channels, ckpt in MODELS:
        if not os.path.isfile(ckpt):
            print(f"Missing checkpoint file: {ckpt}")
            continue

        print(f"Evaluating: {tag}...")

        model = build_model(channels, num_classes)
        
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
            str(len(channels)),
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