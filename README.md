# Stone OCR

## Setup

1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd stone_ocr
   ```
2. Install dependencies (Python 3.11+ recommended):
   - With Poetry:
     ```sh
     poetry install
     ```
   - Or with pip:
     ```sh
     pip install -r requirements.txt
     ```

## Project Structure

- configs/ – YAML configuration files for hyperparameters, model types, training settings, and data augmentation.
- data/ – Datasets for training and evaluation.
- src/stone_ocr/ – Main Python modules: Lightning DataModules, training/evaluation entry points, custom models, utilities.
- lightning_logs/ – Managed by PyTorch Lightning; contains checkpoints, metrics, and TensorBoard logs.
- outputs/ – Generated results: confusion matrices, classification reports, misclassification visualizations, ensemble predictions.
- tests/ – Unit tests for data modules and metrics.
- scripts/ – Example scripts for OCR and data loader visualization.

## Description

Stone OCR is a machine learning project for recognizing inscriptions on stone surfaces. It provides a modular pipeline for preprocessing images, training deep learning models, evaluating results, and analyzing predictions. The project supports multiple model architectures and includes tools for data augmentation, ensemble prediction, and result visualization.
