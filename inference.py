"""
SimCLR Prototype Inference
==========================
Loads a SimCLR encoder, computes class prototypes from the training set,
and runs prototype-based classification with rejection thresholding.

Usage:
    # Step 1 — build the bundle (run once)
    python inference.py --build

    # Step 2 — predict a single image
    python inference.py --image path/to/image.jpg

    # Step 3 — predict a folder of images
    python inference.py --folder path/to/folder/
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available()
                           else "cpu")

SIMCLR_CKPT = Path("simclr_best.pt")
BUNDLE_PATH = Path("simclr_prototype_bundle.pt")
TRAIN_DIR   = Path("rust_dataset/train")
IMG_SIZE    = 224
THRESHOLD   = 0.60     # cosine similarity threshold — lower = more permissive
NUM_WORKERS = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Model ─────────────────────────────────────────────────────────────────────

class SimCLR(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder   = nn.Sequential(*list(base.children())[:-1])
        self.feat_dim  = base.fc.in_features  # 512
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x).squeeze(-1).squeeze(-1)
        z = self.projector(h)
        return h, F.normalize(z, dim=1)

# ── Transforms ────────────────────────────────────────────────────────────────

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Utilities ─────────────────────────────────────────────────────────────────

def l2_normalize(Z, eps=1e-12):
    return Z / (Z.norm(dim=1, keepdim=True) + eps)

def extract_features(model, dataloader):
    all_h, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            h, _   = model(images)
            all_h.append(h.cpu())
            all_labels.append(labels.cpu())
    H = torch.cat(all_h,      dim=0)
    y = torch.cat(all_labels, dim=0)
    return H, y

def compute_prototypes(H, y, class_names):
    protos = []
    for c in range(len(class_names)):
        mask  = (y == c)
        proto = l2_normalize(H[mask]).mean(dim=0)
        protos.append(proto)
    return l2_normalize(torch.stack(protos, dim=0))

# ── Bundle Builder ────────────────────────────────────────────────────────────

def build_bundle():
    print(f"Loading SimCLR checkpoint from {SIMCLR_CKPT} ...")
    model = SimCLR().to(DEVICE)
    model.load_state_dict(torch.load(SIMCLR_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    print(f"Extracting features from {TRAIN_DIR} ...")
    ds = datasets.ImageFolder(TRAIN_DIR, transform=eval_transform)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

    H, y        = extract_features(model, dl)
    class_names = ds.classes
    prototypes  = compute_prototypes(H, y, class_names)

    bundle = {
        "model_state_dict": model.state_dict(),
        "prototypes"      : prototypes,
        "threshold"       : THRESHOLD,
        "class_names"     : class_names,
    }

    torch.save(bundle, BUNDLE_PATH)
    print(f"Bundle saved to {BUNDLE_PATH}")
    print(f"Classes  : {class_names}")
    print(f"Threshold: {THRESHOLD}")

# ── Inference ─────────────────────────────────────────────────────────────────

def load_bundle():
    bundle      = torch.load(BUNDLE_PATH, map_location=DEVICE, weights_only=False)
    model       = SimCLR().to(DEVICE)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    prototypes  = bundle["prototypes"].to(DEVICE)
    threshold   = bundle["threshold"]
    class_names = bundle["class_names"]
    return model, prototypes, threshold, class_names

def predict_single_image(image_path, model, prototypes, threshold, class_names,
                          show=True, save_annotated=True):
    image_path = Path(image_path)
    image      = Image.open(image_path).convert("RGB")
    x          = eval_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        h, _  = model(x)
        h_l2  = l2_normalize(h)
        sims  = h_l2 @ prototypes.T       # [1, num_classes]

        score, pred = sims.max(dim=1)
        score       = float(score.item())
        pred        = int(pred.item())

    nearest_class = class_names[pred]
    accepted      = score >= threshold
    label         = nearest_class if accepted else "REJECT"

    result = {
        "image"        : str(image_path),
        "label"        : label,
        "accepted"     : accepted,
        "nearest_class": nearest_class,
        "score"        : round(score, 4),
        "threshold"    : threshold,
        "similarities" : {cls: round(float(s), 4)
                          for cls, s in zip(class_names, sims.squeeze(0).cpu())},
    }

    if show:
        status = "ACCEPTED" if accepted else "REJECTED"
        print(f"\n{'='*50}")
        print(f"Image     : {image_path.name}")
        print(f"Prediction: {label} ({status})")
        print(f"Score     : {score:.4f}  (threshold={threshold})")
        print(f"Similarities:")
        for cls, sim in result["similarities"].items():
            marker = " ←" if cls == nearest_class else ""
            print(f"  {cls}: {sim:.4f}{marker}")

    if save_annotated and accepted:
        from PIL import ImageDraw
        out_dir   = Path("annotated_results")
        out_dir.mkdir(exist_ok=True)

        annotated = image.copy()
        draw      = ImageDraw.Draw(annotated)
        color     = "red" if label == "CORROSION" else "green"
        draw.rectangle([0, 0, annotated.width, 30], fill=color)
        draw.text((5, 5), f"{label}  {score:.2f}", fill="white")

        out_path = out_dir / image_path.name
        annotated.save(out_path)
        result["annotated_path"] = str(out_path)

    return result

def predict_folder(folder_path, model, prototypes, threshold, class_names):
    folder  = Path(folder_path)
    images  = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    results = []

    print(f"\nRunning inference on {len(images)} images in {folder} ...")
    for img_path in images:
        result = predict_single_image(img_path, model, prototypes,
                                      threshold, class_names,
                                      show=True, save_annotated=True)
        results.append(result)

    accepted = [r for r in results if r["accepted"]]
    rejected = [r for r in results if not r["accepted"]]

    print(f"\n{'='*50}")
    print(f"Summary: {len(images)} images")
    print(f"  Accepted : {len(accepted)}")
    print(f"  Rejected : {len(rejected)}")
    if accepted:
        corrosion = sum(1 for r in accepted if r["label"] == "CORROSION")
        print(f"  CORROSION   : {corrosion}")
        print(f"  NOCORROSION : {len(accepted) - corrosion}")

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build",     action="store_true",
                        help="Build prototype bundle from training set")
    parser.add_argument("--image",     type=str, default=None,
                        help="Path to a single image for inference")
    parser.add_argument("--folder",    type=str, default=None,
                        help="Path to folder of images for inference")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override rejection threshold (default: bundle value)")
    args = parser.parse_args()

    if args.build:
        build_bundle()

    elif args.image or args.folder:
        model, prototypes, threshold, class_names = load_bundle()

        if args.threshold is not None:
            threshold = args.threshold
            print(f"Using override threshold: {threshold}")

        if args.image:
            predict_single_image(args.image, model, prototypes,
                                 threshold, class_names)
        if args.folder:
            predict_folder(args.folder, model, prototypes,
                           threshold, class_names)

    else:
        parser.print_help()
