import os
import time
import cv2
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchmetrics.detection.mean_ap import MeanAveragePrecision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

MODELS = [
    ("CBAM_FPN_r4",  4,  "/models/customcnn_cbam_fpn_r4/customcnn_cbam_fpn_r4_final.pth"),
    ("CBAM_FPN_r8",  8,  "/models/customcnn_cbam_fpn_r8/customcnn_cbam_fpn_r8_final.pth"),
    ("CBAM_FPN_r16", 16, "/models/customcnn_cbam_fpn_r16/customcnn_cbam_fpn_r16_final.pth"),
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

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return img, {"boxes": boxes, "labels": labels}


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

if not os.path.exists(os.path.join(VAL_DIR, "_annotations.csv")):
    print(f"Warning: Annotation file not found at {VAL_DIR}")
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


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        x = x * self.sigmoid_channel(out)

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.conv_spatial(spatial_out)
        x = x * self.sigmoid_spatial(spatial_out)
        return x

class CustomCNNBackboneCBAM(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), reduction=16):
        super().__init__()
        in_ch = 3
        # Use simple naming so BackboneWithFPN works
        for i, out_ch in enumerate(channels):
            stride = 2 if i > 0 else 1
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                CBAM(out_ch, reduction=reduction)
            )
            # Register as self.layer0, self.layer1...
            self.add_module(f"layer{i}", block)
            in_ch = out_ch
        self.out_channels = channels[-1]

    def forward(self, x):

        for name, module in self.named_children():
            x = module(x)
        return x

def build_model(reduction, num_classes):
    channels = (32, 64, 128, 256) # Fixed channels from training script
    
    backbone_base = CustomCNNBackboneCBAM(channels=channels, reduction=reduction)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2'}

    backbone = BackboneWithFPN(
        backbone_base,
        return_layers=return_layers,
        in_channels_list=[channels[1], channels[2], channels[3]],
        out_channels=256
    )

    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=224,
        max_size=224,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
    ).to(device)

    return model


try:
    from thop import profile
    thop_available = True
except Exception:
    thop_available = False
    print("THOP not available. MACs = N/A")


def compute_params(model):
    return sum(p.numel() for p in model.backbone.body.parameters()) / 1e6


def compute_macs(model):
    if not thop_available:
        return None
    dummy = torch.randn(1, 3, 224, 224).to(device)
    macs, _ = profile(model.backbone.body, inputs=(dummy,), verbose=False)
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

        # Build CBAM+FPN Model
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