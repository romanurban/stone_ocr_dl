# visualize_dataloader.py

# ────────────────────────────────────────────────────────────────
# Force multiprocessing to use 'fork' instead of 'spawn' on macOS
import multiprocessing as mp
mp.set_start_method('fork', force=True)
# ────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as TF
import torch
from stone_ocr.datamodules import DefectDataModule

def main():
    # init normally—DefectDataModule has no num_workers kwarg
    dm = DefectDataModule(
        data_dir="data",
        batch_size=4,
    )
    dm.setup()

    # now this train_dataloader can spawn workers safely
    loader = dm.train_dataloader()
    x, y = next(iter(loader))

    print("Batch shape:", x.shape)
    print("Labels:", y)
    print("Class names:", dm.class_names)

    # Visual check
    grid = TF.to_pil_image(torchvision.utils.make_grid(x, nrow=2))
    plt.imshow(grid)
    plt.title("Augumentation sample batch")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
