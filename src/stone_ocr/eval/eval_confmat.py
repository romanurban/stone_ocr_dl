import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from stone_ocr.models.att_unet import AttentionUNetClassifier
from stone_ocr.models.unet import UNetClassifier
from stone_ocr.models.resnet_baseline import ResNetClassifier
from stone_ocr.datamodules import DefectDataModule
from stone_ocr.utils import load_config
from tqdm import tqdm


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confmat.png"):
    cm = confusion_matrix(y_true, y_pred)
    acc = (cm.diagonal().sum() / cm.sum()) * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (acc={acc:.2f}%)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-path", type=str, default="confmat.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dm = DefectDataModule(data_dir=cfg["data_dir"], batch_size=cfg["batch_size"])
    dm.setup("test")
    class_names = dm.class_names

    model_type = cfg.get("model_type", "unet")
    model = None

    if model_type == "unet":
        model = UNetClassifier(
            n_classes=dm.n_classes,
            encoder_name=cfg.get("encoder_name", "resnet34"),
            lr=cfg.get("lr", 1e-3),
            pretrained=False,
            use_unetplusplus=False
        )
    elif model_type == "unetpp":
        model = UNetClassifier(
            n_classes=dm.n_classes,
            encoder_name=cfg.get("encoder_name", "resnet34"),
            lr=cfg.get("lr", 1e-3),
            pretrained=False,
            use_unetplusplus=True
        )
    elif model_type == "att_unet":
        model = AttentionUNetClassifier(
            n_classes=dm.n_classes,
            lr=cfg.get("lr", 1e-3)
        )
    elif model_type == "resnet":
        model = ResNetClassifier(
            n_classes=dm.n_classes,
            lr=cfg.get("lr", 1e-3),
            pretrained=False
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Inference loop
    all_preds, all_labels = [], []
    for x, y in tqdm(dm.test_dataloader(), desc="Predicting"):
        preds = model(x)
        preds = torch.argmax(preds, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

    # Confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path=args.save_path)

    # Print metrics
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))


if __name__ == "__main__":
    main()
