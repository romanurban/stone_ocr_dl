# stone_ocr/model.py
import lightning as L
from stone_ocr.datamodules import DefectDataModule
from stone_ocr.models.unet import UNetClassifier
from stone_ocr.models.att_unet import AttentionUNetClassifier

def build_model(cfg, dm, class_weights):
    model_type = cfg.get("model_type", "unet")
    encoder_name = cfg.get("encoder_name", "resnet34")
    lr = cfg.get("lr", 1e-3)
    pretrained = cfg.get("pretrained", True)
    freeze_encoder = cfg.get("freeze_encoder", False)
    use_focal = cfg.get("use_focal", False)

    if model_type in {"unet", "unetpp"}:
        return UNetClassifier(
            n_classes=dm.n_classes,
            encoder_name=encoder_name,
            lr=lr,
            pretrained=pretrained,
            use_unetplusplus=(model_type == "unetpp"),
            freeze_encoder=freeze_encoder,
            class_weights=class_weights,
            use_focal=use_focal
        )
    elif model_type == "att_unet":
        return AttentionUNetClassifier(
            n_classes=dm.n_classes,
            lr=lr,
            class_weights=class_weights,
            use_focal=use_focal
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")