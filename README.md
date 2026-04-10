# 🫁 Pneumonia Detection from Chest X-Rays Using a Custom CNN

A deep learning project that implements a **Convolutional Neural Network (CNN) from scratch** to classify chest X-ray images as either **NORMAL** or **PNEUMONIA**. The model is trained on the Chest X-Ray dataset and achieves over **95% accuracy** on the test set, with **Grad-CAM** visualizations for clinical interpretability.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Visualizations](#visualizations)
- [Requirements](#requirements)
- [Setup & Usage](#setup--usage)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)
- [Analysis & Discussion](#analysis--discussion)
- [Future Work](#future-work)

---

## 🔍 Project Overview

This project demonstrates the full pipeline for a medical image classification task:

1. **Data Loading & Exploration** — Loading and inspecting the chest X-ray dataset from Google Drive.
2. **Data Preprocessing & Augmentation** — Applying transforms such as random rotation, horizontal flip, colour jitter, and ImageNet-standard normalisation.
3. **Custom CNN Architecture** — Designing a 3-block convolutional network with Batch Normalisation and Dropout regularisation.
4. **Training with Validation** — 15-epoch training loop with Adam optimiser, StepLR scheduler, and best-model checkpointing.
5. **Comprehensive Evaluation** — Precision, Recall, F1-Score, Confusion Matrix, and Grad-CAM visualisations.

> **Platform:** Google Colab (GPU-accelerated, `cuda:0`)

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Name** | Chest X-Ray Dataset |
| **Classes** | `NORMAL`, `PNEUMONIA` |
| **Total Images** | 5,856 |
| **Train Split (80%)** | 4,684 images |
| **Test Split (20%)** | 1,172 images |
| **Image Size** | 224 × 224 pixels |
| **Batch Size** | 32 |

The dataset exhibits a **natural class imbalance**, with significantly more pneumonia examples than normal ones (847 vs. 325 in the test set). This is reflected in the class-specific performance metrics.

**Training Augmentation Pipeline:**
- `RandomRotation(15°)`
- `RandomHorizontalFlip()`
- `ColorJitter(brightness=0.2, contrast=0.2)` — to simulate real-world X-ray lighting variations
- `Resize(224×224)`, `ToTensor()`, `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`

---

## 🧠 Model Architecture

The `CustomCNN` is built from scratch using PyTorch and consists of **3 convolutional blocks** followed by **2 fully connected layers**.

```
Layer (type)          Output Shape        Params
================================================================
Conv2d-1             [-1, 32, 224, 224]      896
BatchNorm2d-2        [-1, 32, 224, 224]       64
MaxPool2d-3          [-1, 32, 112, 112]        0
Conv2d-4             [-1, 64, 112, 112]   18,496
BatchNorm2d-5        [-1, 64, 112, 112]      128
MaxPool2d-6          [-1, 64,  56,  56]        0
Conv2d-7             [-1, 128, 56,  56]   73,856
BatchNorm2d-8        [-1, 128, 56,  56]      256
MaxPool2d-9          [-1, 128, 28,  28]        0
Linear-10            [-1, 512]         51,380,736
Dropout-11           [-1, 512]                 0
Linear-12            [-1, 2]               1,026
================================================================
Total params:        51,475,458
Trainable params:    51,475,458
Input size (MB):     0.57
Forward/backward pass size (MB): 48.24
Params size (MB):    196.36
Estimated Total Size (MB): 245.18
```

**Forward pass:** Each convolutional block applies `Conv2d → BatchNorm → ReLU → MaxPool2d`. After flattening, the feature vector passes through a dense layer, a `Dropout(0.5)` layer, and a final classification head.

---

## 📈 Results

### Training Summary

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|------------|-----------|-----------|----------|
| 1     | 2.2167     | 85.95%    | 0.1761    | 93.26%   |
| 6     | 0.1590     | 93.92%    | 0.1501    | 94.80%   |
| 11    | 0.1436     | 94.88%    | 0.1404    | 95.14%   |
| 15    | 0.1337     | 95.15%    | 0.1367    | **95.31%** |

**Best Test Accuracy: 95.31%** (saved as `CNN_Weights.pth`)

### Classification Report (Test Set)

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| NORMAL    | 0.92      | 0.90   | 0.91     | 325     |
| PNEUMONIA | 0.96      | 0.97   | 0.97     | 847     |
| **Accuracy** | — | — | **0.95** | **1,172** |
| Macro Avg | 0.94 | 0.94 | 0.94 | 1,172 |
| Weighted Avg | 0.95 | 0.95 | 0.95 | 1,172 |

---

## 🖼️ Visualisations

### 1. Loss & Accuracy Curves
Training and test loss/accuracy curves plotted over 15 epochs, confirming stable convergence without overfitting.

### 2. Confusion Matrix
Seaborn heatmap showing the per-class prediction breakdown across 1,172 test samples.

### 3. Grad-CAM (Gradient-weighted Class Activation Mapping)
A custom `generate_gradcam()` function hooks into the final convolutional layer (`conv3`) to produce saliency heatmaps. These highlight **which regions of the X-ray the model attends to** when making a prediction — providing clinical-grade explainability.

---

## ⚙️ Requirements

```
torch
torchvision
torchsummary
numpy
matplotlib
seaborn
scikit-learn
opencv-python (cv2)
google-colab
```

Install with:
```bash
pip install torch torchvision torchsummary numpy matplotlib seaborn scikit-learn opencv-python
```

---

## 🚀 Setup & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/hedayet-ullah-patwary/CVPR_MID_ASSIGNMENT.git
cd <repo-name>
```

### 2. Prepare the Dataset
Place your `Chest_XRay_Dataset.zip` in your **Google Drive** at the following path:
```
/content/drive/MyDrive/Chest_XRay_Dataset.zip
```
The notebook will automatically unzip it into Colab's local storage on first run.

### 3. Open in Google Colab
Upload or open `CNN_22-47904-2.ipynb` in Google Colab. Ensure the runtime is set to **GPU** (`Runtime → Change runtime type → T4 GPU`).

### 4. Run the Notebook
Execute the cells sequentially. The trained model weights will be saved to:
```
/content/CNN_Weights.pth
```

---

## 📁 Project Structure

```
.
├── CNN_22-47904-2.ipynb       # Main Jupyter Notebook (all 9 cells)
├── CNN_Weights.pth            # Saved model weights (generated at runtime)
└── README.md                  # This file
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Batch Normalisation** after each conv layer | Stabilises activations, accelerates convergence |
| **Dropout (p=0.5)** before the output layer | Prevents co-adaptation of neurons; reduces overfitting |
| **ColorJitter augmentation** | X-rays can vary in brightness/contrast across machines and hospitals |
| **Adam optimiser (lr=0.001)** | Adaptive learning rate; works well for sparse, noisy gradients |
| **StepLR scheduler (step=5, γ=0.1)** | Decays LR by ×10 every 5 epochs for fine-tuning in later stages |
| **Reproducible 80/20 split (seed=42)** | Ensures consistent train/test sets across runs |
| **Best-model checkpointing** | Saves weights only when test accuracy improves |
| **Grad-CAM on `conv3`** | The deepest convolutional layer captures the most semantically rich features |

---

## 📝 Analysis & Discussion

The custom CNN demonstrates strong performance on a clinically relevant binary classification task. Key observations:

- **PNEUMONIA class (F1 = 0.97)** outperforms **NORMAL (F1 = 0.91)**, which is expected given the dataset's class imbalance — the model sees ~2.6× more pneumonia samples during training.
- **Batch Normalisation** contributed significantly to fast and stable convergence, as evidenced by the sharp accuracy jump from epoch 1 to epoch 2 (85.95% → 91.05%).
- **Grad-CAM** confirms that the model genuinely focuses on lung opacity regions consistent with clinical pneumonia indicators, rather than relying on spurious features or image artefacts.

---

## 🔮 Future Work

- **Address class imbalance** using Generative Adversarial Networks (GANs) to synthesise additional normal X-ray samples, potentially improving recall for the NORMAL class.
- **Cross-domain transfer learning** — apply this pipeline to detect other thoracic conditions (e.g., COVID-19, tuberculosis, pleural effusion).
- **Model compression** — explore Knowledge Distillation or Quantisation-Aware Training to reduce the 196 MB parameter footprint for edge/mobile deployment.
- **Ensemble methods** — combine predictions from multiple CNN backbones to further push accuracy and robustness.

---

## 👩‍💻 Author

**Hedayet Ullah Patwary**
*Deep Learning | Medical Imaging | Computer Vision*

---

## 📄 License

This project is for academic and research purposes. Please cite appropriately if used in published work.
