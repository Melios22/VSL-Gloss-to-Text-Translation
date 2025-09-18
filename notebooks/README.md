# Notebooks Directory

This directory contains Jupyter notebooks for data processing and model training.

## Notebooks

- **`extract.ipynb`**: Data extraction and preprocessing pipeline
- **`vsl-finetune.ipynb`**: Model fine-tuning experiments and evaluation

## Important Note

The notebooks use `gdown` to download datasets from Google Drive. These download links may become unavailable over time.

**If downloads fail**: You'll need to download the RWTH-PHOENIX-Weather 2014 T dataset yourself and update the file paths in the notebooks accordingly.

## Setup

```bash
pip install gdown jupyter
pip install -r ../requirements.txt
```

Run notebooks sequentially and ensure you have sufficient disk space for datasets and model files.