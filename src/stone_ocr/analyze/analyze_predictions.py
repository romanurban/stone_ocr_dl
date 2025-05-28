import argparse
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from stone_ocr.datamodules import DefectDataModule
from stone_ocr.utils import load_config
from stone_ocr.model import UNetClassifier
from stone_ocr.att_unet import AttentionUNetClassifier

def build_model(cfg, dm):
    model_type = cfg.get("model_type", "unet")
    encoder_name = cfg.get("encoder_name", "resnet34")
    lr = cfg.get("lr", 1e-3)
    pretrained = cfg.get("pretrained", True)

    if model_type == "unet":
        return UNetClassifier(dm.n_classes, encoder_name, lr, pretrained, use_unetplusplus=False)
    elif model_type == "unetpp":
        return UNetClassifier(dm.n_classes, encoder_name, lr, pretrained, use_unetplusplus=True)
    elif model_type == "att_unet":
        return AttentionUNetClassifier(dm.n_classes, lr=lr)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/misclass_analysis")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dm = DefectDataModule(cfg["data_dir"], batch_size=1)
    dm.setup("test")
    class_names = dm.class_names
    test_items = dm.test_set.items
    root_dir = dm.test_set.root

    model = build_model(cfg, dm)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
    model.eval()

    # Output folders
    out_root = Path(args.output_dir)
    (out_root / "correct").mkdir(parents=True, exist_ok=True)
    (out_root / "misclassified").mkdir(parents=True, exist_ok=True)

    for (img_path, label), (x, y) in tqdm(zip(test_items, dm.test_dataloader()), total=len(test_items), desc="Analyzing"):
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        label = y.item()

        true_cls = class_names[label]
        pred_cls = class_names[pred]
        src = root_dir / img_path

        if label == pred:
            dest = out_root / "correct" / f"{true_cls}_{src.name}"
        else:
            dest = out_root / "misclassified" / f"true-{true_cls}_pred-{pred_cls}_{src.name}"

        shutil.copy(src, dest)

    print(f"✅ Done! Misclassified and correct images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
