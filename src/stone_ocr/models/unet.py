import torch
import lightning as L
from segmentation_models_pytorch import Unet, UnetPlusPlus
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.CrossEntropyLoss):
    def __init__(self, alpha=None, gamma=2.0, weight=None):
        super().__init__(weight=weight)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        ce_loss = super().forward(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class UNetClassifier(L.LightningModule):
    def __init__(
        self,
        n_classes,
        encoder_name="resnet34",
        lr=1e-3,
        pretrained=True,
        use_unetplusplus=False,
        freeze_encoder=False,
        class_weights=None,
        use_focal=False
    ):
        super().__init__()
        self.save_hyperparameters()

        # Choose base model
        Net = UnetPlusPlus if use_unetplusplus else Unet
        self.backbone = Net(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            classes=n_classes,  # unused
            activation=None
        )
        self.encoder = self.backbone.encoder

        # Freeze encoder optionally
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Determine feature dimension
        dummy = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            feat = self.encoder(dummy)[-1]
        self.feature_dim = feat.shape[1]

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, n_classes)
        )

        self.lr = lr
        self.use_focal = use_focal
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None

    def forward(self, x):
        feats = self.encoder(x)[-1]         # B x C x H x W
        pooled = self.avgpool(feats)        # B x C x 1 x 1
        flat = pooled.view(x.size(0), -1)   # B x C
        return self.classifier(flat)        # B x n_classes

    def _calculate_loss(self, logits, y):
        if self.use_focal:
            return FocalLoss(weight=self.class_weights)(logits, y)
        else:
            return F.cross_entropy(logits, y, weight=self.class_weights)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test/loss", loss)
        self.log("test/acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
