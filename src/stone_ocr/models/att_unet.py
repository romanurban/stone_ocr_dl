import torch
import torch.nn as nn
import lightning as L
from monai.networks.nets import AttentionUnet
import torchmetrics
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

try:
    from torchmetrics.functional import accuracy
except ImportError:
    accuracy = None

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

class AttentionUNetClassifier(L.LightningModule):
    def __init__(
        self,
        n_classes: int,
        lr=1e-3,
        channels=(64, 128, 256, 512, 1024),
        class_weights=None,
        use_focal_loss=False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = AttentionUnet(
            spatial_dims=2,
            in_channels=3,
            out_channels=channels[-1],
            channels=channels,
            strides=(2, 2, 2, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], n_classes)

        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        else:
            weight_tensor = None

        self.loss_fn = (
            FocalLoss(weight=weight_tensor) if use_focal_loss
            else nn.CrossEntropyLoss(weight=weight_tensor)
        )

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.classifier(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = getattr(self, f"{stage}_acc")
        acc(logits, y)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
