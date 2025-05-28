import torch
from stone_ocr.model import UNetClassifier

# Simulate batch of 4 RGB images, 256x256
dummy_input = torch.randn(4, 3, 256, 256)

# Create model for 8-class classification
model = UNetClassifier(n_classes=8, encoder_name="resnet34")

# Run forward pass
with torch.no_grad():
    out = model(dummy_input)

print("Output shape:", out.shape)
assert out.shape == (4, 8), "Expected output shape (batch_size, n_classes)"
print("✅ Model forward pass works!")
