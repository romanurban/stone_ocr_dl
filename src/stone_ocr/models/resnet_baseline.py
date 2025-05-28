import torch
import torch.nn as nn
import lightning as L
from torchvision.models import resnet34, ResNet34_Weights
import torchmetrics

class ResNetClassifier(L.LightningModule):
    def __init__(self, n_classes, lr=1e-3, pretrained=True):
        super().__init__()
        self.save_hyperparameters()
        
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet34(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, n_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc  = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        return self.backbone(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = getattr(self, f"{stage}_acc")
        acc(logits, y)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/acc",  acc,  prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
