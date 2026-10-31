# Production-Grade Computer Vision Classification System (CNN + Grad-CAM)

An end-to-end Computer Vision classification platform built on **TensorFlow/Keras** using a modern Convolutional Neural Network (CNN) architecture with data augmentation, training callbacks, rigorous evaluation metrics, and visual explainability via **Grad-CAM (Gradient-Weighted Class Activation Mapping)** heatmaps.

---

## 1. Architectural Overview

The convolutional neural network is engineered to process 32x32x3 RGB images (CIFAR-10) through three progressively deeper feature-extraction blocks followed by a regularized dense classification head.

```
Input Image (32x32x3)
        
        

 Block 1: Conv2D (32, 3x3, same) -> BatchNorm -> Conv2D(32) 
 MaxPool2D (2x2) -> Dropout (0.25) 

        
        

 Block 2: Conv2D (64, 3x3, same) -> BatchNorm -> Conv2D(64) 
 MaxPool2D (2x2) -> Dropout (0.25) 

        
        

 Block 3: Conv2D (128, 3x3, same) -> BatchNorm 
 MaxPool2D (2x2) -> Dropout (0.30) 

        
        

 Dense Head: Flatten -> Dense (256, ReLU) -> BatchNorm 
 Dropout (0.50) -> Dense (10, Softmax) 

        
        
Class Probabilities (10 Categories)
```

### Architectural Highlights
- **Spatial Feature Hierarchies**: Conv blocks transition from low-level edges and textures ($32 \to 64$ filters) to complex object semantics ($128$ filters).
- **Internal Covariate Shift Stabilization**: `BatchNormalization` after convolutions stabilizes gradient propagation and accelerates convergence.
- **Overfitting Mitigation**: Combined Spatial Pooling (`MaxPooling2D`), Conv Dropout ($0.25 - 0.30$), and Dense Dropout ($0.50$) prevent co-adaptation.
- **Production Loss & Optimization**: `CategoricalCrossentropy` minimized via `Adam(lr=0.001)`.

---

## 2. Project Structure

```
image_classifier_cnn/
 src/
 __init__.py # Package initialization
 config.py # Hyperparameters, image dimensions, CIFAR-10 class labels, file paths
 dataset.py # Data loader, normalization [0, 1], train/val split, augmentation pipeline
 model.py # Modern multi-block CNN architecture definition
 train.py # Callbacks configuration, training loop, and history curves plot
 evaluate.py # Accuracy, Precision, Recall, F1, classification report, confusion matrix
 explain.py # Grad-CAM heatmap generator and 3-panel saliency overlay
 api/
 __init__.py # API package initialization
 app.py # FastAPI microservice for multipart image upload classification & Grad-CAM
 dashboard/
 __init__.py # Dashboard package initialization
 app.py # Interactive Streamlit studio application
 utils.py # Preprocessing, Grad-CAM generation, and Plotly builders
 models/
 best_cnn_model.keras # Checkpointed best weights saved during training
 artifacts/
 training_history.png # Training vs validation loss & accuracy curves
 confusion_matrix.png # 10-class test confusion matrix heatmap
 gradcam_sample.png # 3-panel visual explainability chart
 main.py # Orchestration script (data -> train -> eval -> test inference -> Grad-CAM)
 requirements.txt # Production dependencies
 README.md # Technical documentation and run manual
```

---

## 3. Data Ingestion & Augmentation

- **Dataset**: CIFAR-10 (60,000 $32 \times 32 \times 3$ color images across 10 categories).
- **Split**: 40,000 training, 10,000 validation, 10,000 holdout testing.
- **Normalization**: Raw byte values $[0, 255]$ scaled to float32 $[0.0, 1.0]$.
- **Augmentation Pipeline**:
  - Horizontal Random Flip
  - Subtle Random Rotation ($\pm 10\%$)
  - Subtle Random Zoom ($\pm 10\%$)
- **High-Throughput Streaming**: `tf.data.Dataset` pipelines with `.shuffle()`, `.batch(64)`, and `.prefetch(tf.data.AUTOTUNE)` for hardware saturation.

---

## 4. Production Training Callbacks

1. **`EarlyStopping`**:
   - Monitored Metric: `val_loss`
   - Patience: 5 epochs
   - Restores best model weights automatically upon trigger.
2. **`ReduceLROnPlateau`**:
   - Monitored Metric: `val_loss`
   - Factor: 0.5 (halves learning rate when plateaus occur)
   - Patience: 3 epochs
   - Minimum LR: $1 \times 10^{-5}$
3. **`ModelCheckpoint`**:
   - Target File: `models/best_cnn_model.keras`
   - Monitored Metric: `val_accuracy`
   - Saves only when validation accuracy improves.

---

## 5. Visual Explainability (Grad-CAM)

Grad-CAM computes the gradient of the predicted class score $y^c$ with respect to the feature activation map $A^k$ of the final convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$

- **Positive Saliency Isolation**: The ReLU operator ensures only features whose activation *increases* the target class score are highlighted.
- **Color Mapping & Superimposition**: The normalized activation matrix is resized to input dimensions ($32 \times 32$), mapped using OpenCV's `COLORMAP_JET`, and superimposed onto the original image at $\alpha = 0.45$ blending opacity.
- **Exported Visualization**: A 3-panel diagnostic graphic (`artifacts/gradcam_sample.png`) displaying:
  1. Input image with ground truth label
  2. Raw Grad-CAM thermal activation heatmap
  3. Superimposed saliency overlay with predicted class and confidence.

---

## 6. Execution Instructions

### A. Environment Installation
```bash
cd image_classifier_cnn
pip install -r requirements.txt
```

### B. End-to-End Orchestration (Train -> Evaluate -> Explain)
```bash
python main.py
```
This automatically downloads CIFAR-10, constructs and trains the CNN, generates training curves, computes test metrics, produces the confusion matrix, and exports the Grad-CAM visualization.

### C. Launching the Inference Microservice
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### D. Testing the API Endpoints

#### 1. Liveness Probe
```bash
curl -X GET http://localhost:8000/health
```

#### 2. Image Classification Inference (Multipart Form)
```bash
curl -X POST http://localhost:8000/classify \
     -F "file=@sample_image.png"
```
**Example Response:**
```json
{
  "filename": "sample_image.png",
  "top_prediction": "frog",
  "top_confidence": 0.8841,
  "top_predictions": [
    {"rank": 1, "class_name": "frog", "probability": 0.8841, "percentage": "88.41%"},
    {"rank": 2, "class_name": "bird", "probability": 0.0523, "percentage": "5.23%"},
    {"rank": 3, "class_name": "cat", "probability": 0.0312, "percentage": "3.12%"}
  ],
  "image_dimensions": [32, 32, 3]
}
```

#### 3. Visual Grad-CAM Attribution Overlay
```bash
curl -X POST http://localhost:8000/explain \
     -F "file=@sample_image.png"
```
Returns prediction score and the superimposed Grad-CAM heatmap rendered as a Base64-encoded PNG data URI.

---

## 7. Interactive Streamlit Studio

Launch the full-featured analytical dashboard:

```bash
streamlit run dashboard/app.py
```

### Dashboard Capabilities
- **Sample Gallery & Custom Uploads**: Test pre-bundled CIFAR-10 test instances or upload arbitrary `.jpg`, `.png`, or `.webp` files.
- **Top-K Confidence Breakdown**: Real-time Plotly horizontal bar chart displaying ranked prediction probabilities.
- **Visual Trio Panel**:
  1. Original image rendered at native resolution.
  2. Isolated Grad-CAM thermal heatmap (Jet colormap).
  3. Superimposed feature activation map blended at 60/40 opacity.
- **Domain-Specific Spatial Diagnostic Callouts**: Automated plain-English explanations highlighting discriminative anatomical and structural contours (fuselage, wheels, wings, ear shape, muzzle, etc.).
- **Live Latency & Health Tracking**: Displays inference latency in milliseconds and alerts when prediction confidence falls below the user-configured warning threshold.


