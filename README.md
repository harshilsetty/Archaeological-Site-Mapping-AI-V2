# Archaeological Site Mapping AI V2

## Overview

This project is a computer vision pipeline for archaeological site mapping using satellite or aerial imagery. It combines object detection, semantic segmentation, and a lightweight Streamlit interface for interactive inference.

The repository contains:

- YOLO-based object detection training and inference
- DeepLabV3+ semantic segmentation training and inference
- COCO-to-mask conversion utilities for segmentation datasets
- A Streamlit app for combined detection and segmentation visualization
- An experimental terrain feature extraction step for erosion-risk estimation

## Tech Stack

### Core Language

- Python

### Machine Learning and Vision

- PyTorch
- Ultralytics YOLO
- segmentation-models-pytorch
- OpenCV
- NumPy
- TorchMetrics
- pycocotools
- Pillow

### UI and Analysis

- Streamlit
- pandas

## Setup

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

## Public Repository Note

This public repository is prepared as a code-first version of the project.

- Large datasets are not tracked in Git
- Trained model weights are not tracked in Git
- Generated run artifacts are not tracked in Git

To fully run training or inference, you will need to place the expected datasets and model weights back into their original local paths or retrain the models.

## Usage

### 1. Generate Segmentation Masks

If the `seg_dataset/*/masks` folders are missing or need to be regenerated from COCO annotations:

```bash
python generate_masks.py
```

### 2. Train YOLO Detection Model

Train the object detection model using the dataset defined in `data.yaml`:

```bash
python train_seg.py
```

This trains a YOLOv8s detector and writes outputs under `runs/detect/`.

### 3. Train DeepLab Segmentation Model

Train the DeepLabV3+ segmentation model:

```bash
python train_deeplab_seg.py
```

This saves the segmentation weights to `deeplab_model.pth`.

### 4. Run Detection on a Sample Image

```bash
python predict.py
```

This expects trained YOLO weights at `runs/detect/yolov8s_archaeology2/weights/best.pt`.

### 5. Run the Combined Demo Pipeline

```bash
python demo_pipeline.py
```

This expects both the YOLO weights and `deeplab_model.pth` to be available locally.

### 6. Launch the Streamlit App

```bash
streamlit run app.py
```

The app supports image upload and displays:

- original image
- YOLO detections
- segmentation overlay
- combined output

### 7. Extract Terrain Features

```bash
python terrain_model/extract_terrain_features.py
```

This creates a terrain-oriented CSV with vegetation ratio, simulated slope, simulated elevation, and erosion-risk labels.

## Main Components

- `app.py`: Streamlit application for image upload and visualization
- `train_seg.py`: YOLOv8 object detection training script
- `train_deeplab_seg.py`: DeepLabV3+ semantic segmentation training script
- `demo_pipeline.py`: local combined pipeline demo
- `predict.py`: single-image detection inference
- `generate_masks.py`: generates segmentation masks from COCO annotations
- `dataset_loader.py`: PyTorch dataset loader for segmentation
- `terrain_model/extract_terrain_features.py`: experimental erosion-risk feature extraction

## Dataset Summary

The repository contains aligned detection and segmentation datasets.

### Split Counts

| Split | Images | Labels | Segmentation Masks |
| --- | ---: | ---: | ---: |
| Train | 648 | 648 | 648 |
| Validation | 61 | 61 | 61 |
| Test | 31 | 31 | 31 |
| Total | 740 | 740 | 740 |

### Detection Dataset

- Location: `dataset/`
- Annotation format: YOLO text labels
- Classes defined in `data.yaml`:
  - `0`: boulders
  - `1`: others
  - `2`: ruins
  - `3`: structures
  - `4`: vegetation

### Segmentation Dataset

- Location: `seg_dataset/`
- Annotation format: COCO JSON plus generated PNG masks
- Model configuration uses 6 classes:
  - `0`: background
  - `1`: boulders
  - `2`: others
  - `3`: ruins
  - `4`: structures
  - `5`: vegetation

### Segmentation Annotation Totals

| Split | Images | COCO Annotations |
| --- | ---: | ---: |
| Train | 648 | 15,760 |
| Validation | 61 | 1,595 |
| Test | 31 | 1,103 |
| Total | 740 | 18,458 |

### Detection Object Counts by Class

| Class | Object Count |
| --- | ---: |
| Vegetation | 10,620 |
| Boulders | 3,876 |
| Ruins | 2,386 |
| Others | 1,346 |
| Structures | 230 |
| Total | 18,458 |

This shows a strong class imbalance, with vegetation dominating the dataset and structures being relatively rare.

### Image Counts by Site Prefix

| Site | Images |
| --- | ---: |
| Hampi | 255 |
| Khajuraho | 238 |
| Pattadakal | 130 |
| Nalanda | 117 |

## Models and Artifacts

### Base Weights Present

These files exist in the current local workspace, but are excluded from the public Git repository:

- `yolov8n.pt`
- `yolov8s.pt`
- `yolo11n.pt`

### Trained Weights Present

These files exist in the current local workspace, but are excluded from the public Git repository:

- `runs/detect/yolov8s_archaeology2/weights/best.pt`
- `runs/detect/train2/weights/best.pt`
- `deeplab_model.pth`

### App Inference Models

The Streamlit app uses:

- YOLO weights from `runs/detect/yolov8s_archaeology2/weights/best.pt`
- DeepLab weights from `deeplab_model.pth`

## Training Configuration

### YOLO Detection Training

From `runs/detect/yolov8s_archaeology2/args.yaml`:

- Model: `yolov8s.pt`
- Task: detection
- Epochs: 80
- Image size: 640
- Batch size: 8
- Device: GPU `0`
- Workers: 8
- Validation enabled during training

### DeepLab Segmentation Training

From `train_deeplab_seg.py`:

- Architecture: DeepLabV3+
- Encoder: ResNet34
- Encoder weights: ImageNet
- Input size: `512 x 512`
- Batch size: 4
- Optimizer: Adam
- Learning rate: `1e-4`
- Loss: CrossEntropyLoss
- Metrics: IoU and Dice
- Epochs in script: 10

## Saved Detection Results

From `runs/detect/yolov8s_archaeology2/results.csv`:

- Best `mAP50-95`: `0.58673` at epoch `78`
- Best `mAP50`: `0.83206` at epoch `80`
- Final epoch `80` metrics:
  - Precision: `0.89978`
  - Recall: `0.72010`
  - `mAP50`: `0.83206`
  - `mAP50-95`: `0.58611`

## Project Structure Notes

- `runs/detect/` contains training and prediction outputs from multiple YOLO runs
- `seg_dataset/` includes COCO annotations and generated masks for all splits
- `terrain_model/` currently contains one experimental script for erosion-related feature extraction

## Observations

- The project is a hybrid detection plus segmentation system rather than a single-model repository.
- Detection and segmentation datasets are aligned and use the same image counts across splits.
- The local workspace is inference-ready because the trained YOLO and DeepLab weights referenced by the app exist.
- The class distribution is imbalanced, which may affect model performance on underrepresented classes such as structures.
- The repository now includes `requirements.txt`, but package versions are estimated from imports rather than exported from a known working environment.
- The segmentation COCO metadata appears to contain an unusual category naming entry for class `0`, while the code clearly treats class `0` as background.

## Recommended Next Improvements

- Add a `pyproject.toml` or fully pinned environment export
- Add a proper training and inference usage guide
- Document expected hardware requirements for YOLO and DeepLab training
- Clean up naming so detection and segmentation scripts are easier to distinguish
- Add segmentation evaluation logs or saved metrics similar to YOLO training outputs 


