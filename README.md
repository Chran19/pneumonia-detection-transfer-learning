# Pneumonia Detection Using Transfer Learning with VGG16

GitHub Python TensorFlow Keras Accuracy

Repository: [pneumonia-detection-transfer-learning](https://github.com/Chran19/pneumonia-detection-transfer-learning)

## Project Overview

This project implements the approach described in the research paper **"CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning"** by Rajpurkar et al. (2017). We use **transfer learning with VGG16** (pre-trained on ImageNet) to classify pediatric chest X-ray images as **Normal** or **Pneumonia**. The project demonstrates a complete deep learning pipeline including data augmentation, model training, evaluation, and performance comparison with the original research paper.

## Team Members

| Name               | Roll Number  |
| ------------------ | ------------ |
| Yashwardhan Jangid | 202301100007 |
| Shreyash Kumbhar   | 202301100032 |
| Ranjeet Choudhary  | 202301100046 |
| Rishabh Rai        | 202301100047 |

## Project Structure

```
Assignment-2/
├── Lab_Assignment_2_Submission.ipynb        # Main Jupyter notebook (all tasks)
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
├── pneumonia_vgg16_model.keras             # Saved model weights
├── images/                                  # Saved plots and visualizations
│   ├── sample_xray_images.png
│   ├── class_distribution.png
│   ├── augmented_images.png
│   ├── feature_maps.png
│   ├── sample_predictions.png
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── performance_comparison.png
└── chest_xray/                              # Dataset directory
    ├── train/
    │   ├── NORMAL/       (1,341 images)
    │   └── PNEUMONIA/    (3,875 images)
    ├── val/
    │   ├── NORMAL/       (8 images)
    │   └── PNEUMONIA/    (8 images)
    └── test/
        ├── NORMAL/       (234 images)
        └── PNEUMONIA/    (390 images)
```

## Neural Network Architecture

```
Input Layer (224×224×3 images)
        ↓
VGG16 Base Model (Pre-trained, block5 fine-tuned)
        ↓
GlobalAveragePooling2D (7×7×512 → 512)
        ↓
BatchNormalization
        ↓
Dense Layer (512, ReLU activation)
        ↓
Dropout (0.4)
        ↓
Dense Layer (256, ReLU activation)
        ↓
BatchNormalization
        ↓
Dropout (0.3)
        ↓
Output Layer (1, Sigmoid activation)
```

### Features Used

- Image Input: 224×224 pixels, RGB channels (3)
- Pre-trained Weights: ImageNet
- Fine-tuning: VGG16 block5 layers unfrozen
- Total Parameters: 15,112,001 (57.65 MB)
- Trainable Parameters: 7,475,201 (28.52 MB)
- Non-trainable Parameters: 7,636,800 (29.13 MB)

## Results

| Metric            | Value  |
| ----------------- | ------ |
| Training Accuracy | 97.50% |
| Testing Accuracy  | 92.95% |
| Precision         | 92.61% |
| Recall            | 96.41% |
| F1-Score          | 94.47% |
| Total Epochs      | 20     |
| Learning Rate     | 0.0001 |
| Optimizer         | Adam   |
| Batch Size        | 32     |

### Performance Comparison with Research Paper

| Metric    | CheXNet (Rajpurkar et al.) | Our Implementation | Difference |
| --------- | -------------------------- | ------------------ | ---------- |
| Accuracy  | 92.80%                     | 92.95%             | +0.15%     |
| Precision | 93.20%                     | 92.61%             | -0.59%     |
| Recall    | 93.90%                     | 96.41%             | +2.51%     |
| F1-Score  | 93.50%                     | 94.47%             | +0.97%     |

### Classification Performance

| Class Type | Precision | Recall | F1-Score |
| ---------- | --------- | ------ | -------- |
| NORMAL     | 0.94      | 0.87   | 0.90     |
| PNEUMONIA  | 0.93      | 0.96   | 0.94     |

### Confusion Matrix Summary

- **True Negatives (NORMAL correctly classified):** 204
- **True Positives (PNEUMONIA correctly classified):** 376
- **False Positives (NORMAL misclassified as PNEUMONIA):** 30
- **False Negatives (PNEUMONIA misclassified as NORMAL):** 14

### Visualizations

| | |
|---|---|
| ![Class Distribution](images/class_distribution.png) | ![Confusion Matrix](images/confusion_matrix.png) |
| Class Distribution | Confusion Matrix |
| ![Training History](images/training_history.png) | ![Performance Comparison](images/performance_comparison.png) |
| Training History | Performance Comparison |
| ![Sample Predictions](images/sample_predictions.png) | ![Feature Maps](images/feature_maps.png) |
| Sample Predictions | Feature Maps |

## Getting Started

### Prerequisites

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python Pillow h5py
```

### Running the Notebook

1. Download the dataset from [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Place the `chest_xray/` folder in the project directory
3. Open `Lab_Assignment_2_Submission.ipynb` in Jupyter Notebook or VS Code
4. Run all cells sequentially

### Using the Saved Model

```python
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('pneumonia_vgg16_model.keras')

# Make predictions on new images
from PIL import Image
import numpy as np

img = Image.open('path/to/xray.jpg').resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
class_name = 'PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL'
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

print(f"Class: {class_name}, Confidence: {confidence:.4f}")
```

## Key Features

- **Transfer Learning Implementation** - Leveraging VGG16 pre-trained on ImageNet
- **Data Augmentation** - Rotation, shifts, zoom, flip, and brightness adjustments
- **Feature Visualization** - Visualization of feature maps from VGG16 layers
- **Multi-loss Tracking** - Monitoring training and validation accuracy/loss
- **Callbacks Integration** - EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Comprehensive Evaluation** - Confusion matrix, classification report, and metrics
- **Model Persistence** - Save and load trained model in Keras format
- **Research Comparison** - Side-by-side comparison with Rajpurkar et al. (CheXNet) paper results

## Dataset Description

The dataset contains 5,856 pediatric chest X-ray images with the following attributes:

| Attribute          | Details                                                                          |
| ------------------ | -------------------------------------------------------------------------------- |
| Image Format       | JPEG                                                                             |
| Image Size         | 224×224 pixels                                                                   |
| Color Channels     | RGB (3 channels)                                                                 |
| Classes            | 2 (NORMAL, PNEUMONIA)                                                            |
| Total Samples      | 5,856                                                                            |
| Data Splits        | Train: 5,216 (89.1%), Val: 16 (0.3%), Test: 624 (10.6%)                          |
| Class Distribution | NORMAL: 1,583 (27.0%), PNEUMONIA: 4,273 (73.0%)                                  |
| Source             | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |

## Methodology

### Task 1: Data Exploration & Preprocessing

- Load dataset from organized folder structure
- Analyze class distribution and data imbalance
- Implement ImageDataGenerator for data augmentation
- Visualize sample images and augmentation effects

### Task 2: Model Development & Training

- Load VGG16 pre-trained on ImageNet
- Fine-tune VGG16 block5 convolutional layers
- Build custom classification head with BatchNormalization and Dropout
- Apply class weights to handle data imbalance (NORMAL: 1.94, PNEUMONIA: 0.67)
- Compile model with Adam optimizer (lr=1e-4) and binary cross-entropy loss
- Train for 20 epochs with callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Save trained model in Keras format

### Task 3: Evaluation & Comparison

- Generate predictions on test set
- Compute classification metrics (accuracy, precision, recall, F1-score)
- Create confusion matrix with heatmap visualization
- Plot training history (accuracy and loss curves)
- Compare results with original research paper
- Analyze performance differences and overfitting indicators

## Research Paper Reference

- **Title:** CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning
- **Authors:** Pranav Rajpurkar, Jeremy Irvin, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Curtis Langlotz, et al.
- **Year:** 2017
- **arXiv:** https://arxiv.org/abs/1711.05225
- **DOI:** https://doi.org/10.48550/arXiv.1711.05225

## License

This project is developed for educational purposes as part of a neural networks lab assignment.

---

**Developed by Team**

- Yashwardhan Jangid (202301100007)
- Shreyash Kumbhar (202301100032)
- Ranjeet Choudhary (202301100046)
- Rishabh Rai (202301100047)
