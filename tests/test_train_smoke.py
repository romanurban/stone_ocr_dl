from stone_ocr.utils import load_config
from stone_ocr.models.model import UNetClassifier
from stone_ocr.datamodules import DefectDataModule
import lightning as L

def test_training_loop_runs_one_epoch():
    cfg = load_config("configs/debug.yaml")
    dm = DefectDataModule(cfg["data_dir"], batch_size=cfg["batch_size"])
    model = UNetClassifier(n_classes=dm.n_classes, encoder_name=cfg["encoder_name"])

    trainer = L.Trainer(max_epochs=1, fast_dev_run=True)
    trainer.fit(model, datamodule=dm)
    print("✅ Training smoke test passed")

# manually call the function so it runs when you use python -m
if __name__ == "__main__":
    test_training_loop_runs_one_epoch()
