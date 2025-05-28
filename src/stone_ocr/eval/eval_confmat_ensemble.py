# stone_ocr/eval_confmat_ensemble.py
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from stone_ocr.datamodules import DefectDataModule
from stone_ocr.models.unet import UNetClassifier
from stone_ocr.models.att_unet import AttentionUNetClassifier
from stone_ocr.models.resnet_baseline import ResNetClassifier
from stone_ocr.utils import load_config
from tqdm import tqdm


def load_model(cfg_path, ckpt_path, n_classes):
    """
    Load a model (ResNet, U-Net, U-Net++ or Attention U-Net) from a Lightning checkpoint.
    Supports optional strict=False to skip mismatched keys when loading state_dict.
    """
    cfg = load_config(cfg_path)
    model_type = cfg.get("model_type", "unet")
    encoder_name = cfg.get("encoder_name", "resnet34")
    lr = cfg.get("lr", 1e-3)
    class_weights = None
    if cfg.get("class_weights"):
        class_weights = torch.tensor(cfg.get("class_weights"))

    # Unified U-Net / U-Net++ loader
    if model_type in ("unet", "unetpp"):
        use_unetplusplus = cfg.get("use_unetplusplus", model_type == "unetpp")
        model = UNetClassifier.load_from_checkpoint(
            ckpt_path,
            n_classes=n_classes,
            encoder_name=encoder_name,
            lr=lr,
            pretrained=False,
            use_unetplusplus=use_unetplusplus,
            class_weights=class_weights,
            strict=False  # allow missing/unexpected keys
        )

    # Attention U-Net
    elif model_type == "att_unet":
        model = AttentionUNetClassifier.load_from_checkpoint(
            ckpt_path,
            n_classes=n_classes,
            lr=lr,
            class_weights=class_weights,
            strict=False
        )

    # ResNet baseline
    elif model_type == "resnet":
        model = ResNetClassifier.load_from_checkpoint(
            ckpt_path,
            n_classes=n_classes,
            lr=lr,
            pretrained=False
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    return model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, help="List of config .yaml files")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint .ckpt files")
    parser.add_argument("--save-path", type=str, default="ensemble_confmat.png")
    args = parser.parse_args()

    assert len(args.configs) == len(args.checkpoints), "Mismatch in number of configs and checkpoints!"

    # Initialize datamodule from the first config
    dm_cfg = load_config(args.configs[0])
    dm = DefectDataModule(data_dir=dm_cfg["data_dir"], batch_size=dm_cfg["batch_size"])
    dm.setup("test")
    class_names = dm.class_names

    # Load models
    models = []
    for cfg_path, ckpt_path in zip(args.configs, args.checkpoints):
        cfg = load_config(cfg_path)
        print(f"\n📄 Loaded config from: {cfg_path}")
        for key, value in cfg.items():
            print(f"  {key}: {value}")
        model = load_model(cfg_path, ckpt_path, dm.n_classes)
        models.append(model)

    # Ensemble predictions
    all_preds, all_labels = [], []
    for x, y in tqdm(dm.test_dataloader(), desc="Ensemble prediction"):
        logits_list = [m(x).softmax(dim=1) for m in models]
        avg_logits = torch.stack(logits_list).mean(dim=0)
        preds = torch.argmax(avg_logits, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

    # Confusion matrix and report
    cm = confusion_matrix(all_labels, all_preds)
    acc = (cm.diagonal().sum() / cm.sum()) * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Ensemble Confusion Matrix (acc={acc:.2f}%)")
    plt.tight_layout()
    plt.savefig(args.save_path)
    print(f"Confusion matrix saved to {args.save_path}")

    print("\nEnsemble Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))


if __name__ == "__main__":
    main()
