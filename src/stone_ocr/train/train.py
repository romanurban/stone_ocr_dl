import lightning as L
import torch
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from stone_ocr.datamodules import DefectDataModule
from stone_ocr.models.unet import UNetClassifier
from stone_ocr.models.att_unet import AttentionUNetClassifier
from stone_ocr.utils import load_config
from stone_ocr.models.resnet_baseline import ResNetClassifier


def build_model(cfg, dm):
    model_type = cfg.get("model_type", "unet")
    encoder_name = cfg.get("encoder_name", "resnet34")
    lr = cfg.get("lr", 1e-3)
    pretrained = cfg.get("pretrained", True)
    class_weights = torch.tensor(cfg.get("class_weights", [])) if cfg.get("class_weights") else None

    if model_type == "unet":
        return UNetClassifier(
            n_classes=dm.n_classes,
            encoder_name=encoder_name,
            lr=lr,
            pretrained=pretrained,
            use_unetplusplus=False,
            class_weights=class_weights
        )
    elif model_type == "unetpp":
        return UNetClassifier(
            n_classes=dm.n_classes,
            encoder_name=encoder_name,
            lr=lr,
            pretrained=pretrained,
            use_unetplusplus=True,
            class_weights=class_weights
        )
    elif model_type == "att_unet":
        return AttentionUNetClassifier(
            n_classes=dm.n_classes,
            lr=lr,
            class_weights=class_weights
        )
    elif model_type == "resnet":
        return ResNetClassifier(
            n_classes=dm.n_classes,
            lr=lr,
            pretrained=False
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to .yaml config")
    parser.add_argument("--test-only", action="store_true", help="Run only evaluation using best model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dm = DefectDataModule(data_dir=cfg["data_dir"], batch_size=cfg["batch_size"])

    if cfg.get("use_class_weights", False):
        print("Computing class weights from training set...")
        class_counts = torch.zeros(dm.n_classes)
        for _, label in dm.train_set:
            class_counts[label] += 1
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * dm.n_classes
        cfg["class_weights"] = class_weights.tolist()

    model = build_model(cfg, dm)

    checkpoint_cb = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        filename="{epoch:02d}-{val_acc:.2f}",
    )

    logger = CSVLogger("logs", name=cfg.get("experiment_name", "default"))

    trainer = L.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="auto", 
        devices="auto",
        log_every_n_steps=1,
        precision=cfg.get("precision", "32-true"),
        callbacks=[checkpoint_cb],
        logger=logger
    )

    if not args.test_only:
        trainer.fit(model, datamodule=dm)

    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        print(f"Loading best model from: {best_ckpt}")
        model_type = cfg.get("model_type", "unet")
        class_weights = torch.tensor(cfg.get("class_weights", [])) if cfg.get("class_weights") else None

        if model_type in {"unet", "unetpp"}:
            model = UNetClassifier.load_from_checkpoint(
                best_ckpt,
                n_classes=dm.n_classes,
                encoder_name=cfg.get("encoder_name", "resnet34"),
                lr=cfg.get("lr", 1e-3),
                pretrained=False,
                use_unetplusplus=(model_type == "unetpp"),
                class_weights=class_weights
            )
        elif model_type == "att_unet":
            model = AttentionUNetClassifier.load_from_checkpoint(
                best_ckpt,
                n_classes=dm.n_classes,
                lr=cfg.get("lr", 1e-3),
                class_weights=class_weights
            )
        elif model_type == "resnet":
            model = ResNetClassifier.load_from_checkpoint(
                best_ckpt,
                n_classes=dm.n_classes,
                lr=cfg.get("lr", 1e-3),
                pretrained=False
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()