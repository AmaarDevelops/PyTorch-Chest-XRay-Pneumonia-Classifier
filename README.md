

# 👨‍⚕️ PyTorch Chest X-Ray Pneumonia Classifier: Multi-Stage Fine-Tuning

This project implements a robust deep learning solution for binary classification of Chest X-Ray images (Pneumonia vs. Normal). It utilizes **Transfer Learning** with a pre-trained **ResNet18** model and employs an advanced, **three-phase gradual fine-tuning strategy** to achieve highly optimized performance, particularly suited for medical imaging data.


---

## ⚙️ Project Setup and Prerequisites

### 1. Data
The model is trained on the **Chest X-Ray Images (Pneumonia)** dataset, split into `train`, `val`, and `test` directories.

### 2. Dependencies
Install the required Python packages:

pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pytorch-grad-cam opencv-python


### 3\. Execution Environment

This script is highly optimized for **GPU usage**. Training should be executed on a GPU-enabled environment (e.g., Google Colab, Kaggle, or a local machine with CUDA) to reduce runtime from hours to minutes.

-----

## 🧠 Methodology: Three-Phase Gradual Fine-Tuning

The training process is structured into three distinct phases to prevent catastrophic forgetting and ensure optimal weight tuning. Each phase utilizes **Early Stopping** and **Model Checkpointing** based on validation loss, and a **StepLR Scheduler** for controlled learning rate decay.

### Phase 1: Feature Extraction (Frozen Backbone)

  * **Goal:** Quickly train the new, randomly initialized **Fully Connected (FC) layer**.
  * **Training:** Only the final `model.fc` layer is trainable. The entire ResNet18 backbone is frozen.
  * **Learning Rate:** High ($\approx 0.001$).

### Phase 2: Shallow Fine-Tuning (Unfreeze Last Block)

  * **Goal:** Tune the highest-level feature extractor (`layer4`) for domain-specific patterns (X-rays).
  * **Training:** The final residual block (`model.layer4`) and the FC layer are trainable.
  * **Learning Rate:** Reduced ($\approx 0.0005$).

### Phase 3: Deep Fine-Tuning (Unfreeze Entire Network)

  * **Goal:** Micro-adjust all network weights for maximum performance.
  * **Training:** The entire ResNet18 network (all layers) is trainable.
  * **Learning Rate:** Very low ($\approx 0.0001$).

-----

## 📈 Results and Evaluation

The final model weights—the single best checkpoint saved across all three phases—are loaded for the final, unbiased evaluation on the unseen `test` dataset.




## 📂 Model Usage and Files

  * `Nn.py`: The main Python script containing the entire training and evaluation pipeline.
  * `X_ray_classifier.pth`: The final, optimized `state_dict` (weights) of the best-performing model checkpoint.
  * `class_to_index.pth`: A mapping file required for deployment (`{'NORMAL': 0, 'PNEUMONIA': 1}`).

Dataset link :- https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
