# TransXNet-DMC

A lightweight multimodal brain tumor classification project based on paired CT/MRI images.

This release package is prepared for open-source publishing and includes only essential source code and scripts.

## Highlights

- Multimodal training script for paired CT and MRI images.
- Clean model folder with staged TransXNet variants renamed for clarity.
- Portable training script without machine-specific paths.

## Renamed model files

The following files were renamed from intermediate experiment names:

- `models/transxnet_mca.py` (old: `transxnetg.py`)
- `models/transxnet_mca_mudd.py` (old: `transxnetgg.py`)
- `models/transxnet_dmc.py` (old: `transxnetggg.py`)

## Repository structure

```text
TransXNet-DMC/
  fl.py
  run.sh
  README.md
  requirements.txt
  .gitignore
  models/
    transxnet.py
    transxnet_mca.py
    transxnet_mca_mudd.py
    transxnet_dmc.py
    repvit.py
    edgenextsmall.py
    ...
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you use conda, create and activate your own environment first, then install dependencies from `requirements.txt`.

## Dataset

This code expects the dataset under:

```text
./Dataset/
  Brain Tumor CT scan Images/
    Healthy/
    Tumor/
  Brain Tumor MRI images/
    Healthy/
    Tumor/
```

Dataset source:

- https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri

## Select backbone

Edit `Config.backbone_name` in `fl.py`.

Available options:

- `repvit` (default)
- `edgenext`
- `transxnet_base`
- `transxnet_mca`
- `transxnet_mca_mudd`
- `transxnet_dmc`

## Train

Option 1:

```bash
python fl.py
```

Option 2:

```bash
bash run.sh
```

You can override Python executable when using `run.sh`:

```bash
PYTHON_BIN=python3 bash run.sh
```
