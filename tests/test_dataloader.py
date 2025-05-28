from stone_ocr.datamodules import DefectDataModule
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as TF

dm = DefectDataModule(data_dir="data", batch_size=4)
dm.setup()

loader = dm.train_dataloader()
x, y = next(iter(loader))

print("Batch shape:", x.shape)
print("Labels:", y)
print("Class names:", dm.class_names)

# Visual check
grid = TF.to_pil_image(torchvision.utils.make_grid(x, nrow=2))
plt.imshow(grid)
plt.title("Sample batch")
plt.axis("off")
plt.show()
