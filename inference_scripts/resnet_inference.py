import os
import time
import cv2
import torch
import torch.nn as nn
import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


MODEL_PATH = "/models/resnet50/resnet50_tiled_final.pth"

MODELS = [
    ("ResNet50_Tiled", "ResNet50", MODEL_PATH),
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


def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None) 
  
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model.to(device)


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
    headers = ["Model", "Backbone", "mAP@50", "mAP@50:95", "FPS", "Bkbn Params(M)", "Bkbn MACs(G)"]
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
    rows = []

    num_classes = len(unique_classes) + 1

    for tag, backbone_name, ckpt in MODELS:
        if not os.path.isfile(ckpt):
            print(f"Missing checkpoint file: {ckpt}")
            continue

        print(f"Evaluating: {tag}...")

        model = build_model(num_classes)

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
            backbone_name,
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