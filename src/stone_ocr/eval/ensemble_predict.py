import argparse
import torch
import numpy as np
from tqdm import tqdm
from stone_ocr.models.model import UNetClassifier
from stone_ocr.models.att_unet import AttentionUNetClassifier
from stone_ocr.datamodules import DefectDataModule
from stone_ocr.utils import load_config
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def load_model(cfg, ckpt_path, model_type, n_classes):
    if model_type in {"unet", "unetpp"}:
        model = UNetClassifier(
            n_classes=n_classes,
            encoder_name=cfg.get("encoder_name", "resnet34"),
            lr=cfg.get("lr", 1e-3),
            pretrained=False,
            use_unetplusplus=(model_type == "unetpp")
        )
    elif model_type == "att_unet":
        model = AttentionUNetClassifier(
            n_classes=n_classes,
            lr=cfg.get("lr", 1e-3)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoints", nargs='+', required=True, help="List of model checkpoints")
    parser.add_argument("--types", nargs='+', required=True, help="Model types: unet / unetpp / att_unet")
    parser.add_argument("--save-confmat", type=str, default="confmat_ensemble.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dm = DefectDataModule(data_dir=cfg["data_dir"], batch_size=cfg["batch_size"])
    dm.setup("test")
    class_names = dm.class_names

    assert len(args.checkpoints) == len(args.types), "Mismatch between number of checkpoints and model types"

    models = [
        load_model(cfg, ckpt_path, model_type, dm.n_classes)
        for ckpt_path, model_type in zip(args.checkpoints, args.types)
    ]

    all_preds, all_labels = [], []

    for x, y in tqdm(dm.test_dataloader(), desc="Ensemble prediction"):
        logits_list = [model(x) for model in models]
        avg_logits = torch.stack(logits_list).mean(dim=0)
        preds = torch.argmax(avg_logits, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    acc = (cm.diagonal().sum() / cm.sum()) * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Ensemble Confusion Matrix (acc={acc:.2f}%)")
    plt.tight_layout()
    plt.savefig(args.save_confmat)
    print(f"Confusion matrix saved to {args.save_confmat}")


if __name__ == "__main__":
    main()
