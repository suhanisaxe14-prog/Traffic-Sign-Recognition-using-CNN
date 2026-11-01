### 🧠 Traffic Sign Recognition on GTSRB: ANN vs Classical ML (HOG + SVM) vs CNN

**A deep-learning computer-vision project that recognises 43 types of German traffic signs and proves, with a fair three-way experiment, why a CNN is the right tool for the job.** 🚀

![Model comparison](docs/model_comparison.png)

---

## 🎯 About the Project

Traffic-sign recognition is a core perception task for driver-assistance systems and autonomous vehicles. A misread sign (a Stop taken for a speed limit, say) is a safety problem, so the classifier must be accurate across **all** classes, including rare ones.

SignSense solves the **same problem three different ways**, on the **same data split**, and compares them:

1. **ANN**: a fully connected network that sees raw pixels.
2. **Classical ML**: hand-crafted **HOG** (shape) + **HSV colour** features classified by an **SVM**.
3. **CNN**: a convolutional network that learns its own features. This is the final model.

The project is built only with core fundamentals (ANN, CNN, classical ML), with no transfer learning or fine-tuning.

## 🏆 Results at a Glance

| Model | Input | Test accuracy |
|---|---|---|
| ANN | Raw pixels | **83.17 %** |
| HOG + colour + SVM (RBF) | Hand-crafted features | **89.64 %** |
| **CNN** | Raw pixels, learned features | **96.01 %** |

CNN validation accuracy: **98.33 %** · wrong predictions: **504 / 12,630** · macro F1: **0.94**

## ✨ Key Features

- 🔀 **Leak-free validation**: split by *sign track*, not by random image (details below)
- 🧪 **Three-model comparison** on identical train / validation / test data
- 🔄 **Data augmentation** built into the CNN (rotation, zoom, translation)
- 🏗️ **Batch normalisation + dropout + early stopping** for stable, regularised training
- 🔍 **HOG visualisation** to show what the classical model "sees"
- 🎛️ **SVM hyper-parameter grid search** and **feature ablation**
- 📊 **Full evaluation**: per-class report, confusion matrices, weakest classes, misclassified images
- 🥇 **Top-3 predictions** with confidence for any test image
- 💾 **Saved models**: `gtsrb_cnn.keras` and `traffic_sign_svm.joblib`

## 🛠️ Technologies Used

| Technology | Role in the project |
|---|---|
| 🐍 Python | Core language |
| 🧠 Keras 3 / TensorFlow | ANN and CNN |
| 🤖 scikit-learn | SVM, PCA, scaling, metrics, `GroupShuffleSplit` |
| 🖼️ scikit-image | HOG, CLAHE, HSV conversion |
| 🔢 NumPy & Pandas | Data handling |
| 📊 Matplotlib | All charts |
| 🗂️ Pillow, tqdm, joblib, kagglehub | Image loading, progress, model saving, dataset download |
| 📓 Jupyter Notebook | Development |

## 🗂️ Dataset: GTSRB

**German Traffic Sign Recognition Benchmark** (IJCNN 2011), downloaded from Kaggle with `kagglehub`:
[`meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

| Property | Value |
|---|---|
| Classes | 43 |
| Training images | 39,209 (1,307 sign "tracks") |
| Test images | 12,630 |
| Image size used | 32 × 32 × 3 (RGB) |
| Class balance | **Imbalanced**: 210 to 2,250 images per class |


### Class distribution
Some classes have more than 10× the images of others, so overall accuracy alone can hide weak classes.


### 🔒 Avoiding data leakage
Training file names look like `<class>_<track>_<frame>.png`. Frames of one physical sign are near-identical, so a random split would leak almost the same picture into validation and inflate the score. SignSense splits **by track** with `GroupShuffleSplit` and verifies it with an `assert`:

| Split | Images |
|---|---|
| Train | 31,349 |
| Validation | 7,860 (no track shared with train) |
| Test | 12,630 (official test set, never used for tuning) |

## ⚙️ How It Works

🖼️ **GTSRB image**
⬇️
📐 **Resize to 32×32 RGB, scale pixels to [0, 1]**
⬇️
✂️ **Track-wise train / validation split**
⬇️
🔀 **Three pipelines: ANN · HOG + SVM · CNN**
⬇️
📊 **Evaluate all three on the same held-out test set**
⬇️
🥇 **Select the CNN and save it**

## 🧠 Models

### 1️⃣ ANN (baseline)
```
Flatten → Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dense(43, softmax)
```
About **825 K parameters**, Adam, 10 epochs. Flattening throws away the 2-D layout of the pixels, which is why it is the weakest.


### 2️⃣ Classical ML: HOG + colour + SVM

| Step | Detail |
|---|---|
| Preprocessing | Grayscale + **CLAHE** (fixes uneven lighting) |
| Shape features | **HOG**: 9 orientations, 4×4 cells, 2×2 blocks → 1,764 features |
| Colour features | HSV histogram, 8 bins × 3 channels → 24 features |
| Scaling + PCA | `StandardScaler` → PCA keeping 95 % variance: 1,788 → **505** features |
| Classifier | **SVM, RBF kernel**, `class_weight="balanced"` |
| Tuning | Grid over `C` ∈ {1, 10, 50} × `gamma` ∈ {0.5, 1, 2}× scale, on the track-wise validation set |


**Hyper-parameter tuning** (best: `C = 10`, `gamma = scale × 0.5`, validation 90.57 %):


**Feature ablation**: shape carries almost all the signal, colour alone is nearly useless:

| Features | Validation accuracy |
|---|---|
| Colour only | 13.8 % |
| HOG only | 89.7 % |
| HOG + colour | 90.0 % |


**Kernel check**: RBF **90.6 %** vs linear **86.3 %**, so the classes are not linearly separable in HOG space.

### 3️⃣ CNN (final model)
```
Input 32×32×3
 → Augmentation (RandomRotation 0.08 · RandomZoom 0.10 · RandomTranslation 0.10)   [training only]
 → Conv2D(32)  → BatchNorm → MaxPool
 → Conv2D(64)  → BatchNorm → MaxPool
 → Conv2D(128) → BatchNorm → MaxPool
 → Flatten → Dense(128, ReLU) → Dropout(0.4) → Dense(43, softmax)
```
About **362 K parameters**, less than half the ANN, yet far more accurate. Adam (lr 1e-3), up to 30 epochs, **EarlyStopping** on `val_loss` (patience 5, best weights restored).

| Component | Why it is there |
|---|---|
| Conv layers | Learn edges → shapes → sign parts, with weights shared across the image |
| BatchNorm | Faster, more stable training |
| MaxPooling | Tolerates small shifts, fewer parameters |
| Augmentation | Signs appear at different angles, scales, positions |
| Dropout | Reduces overfitting in the dense layer |
| EarlyStopping | Stops at the best epoch |


**CNN training curves:**

![CNN curves](docs/cnn_curves.png)

## 📈 Model Evaluation

### Overall comparison

| Model | Test accuracy | Misclassified | Notes |
|---|---|---|---|
| ANN | 83.17 % | n/a | No spatial awareness |
| HOG + colour + SVM | 89.64 % | 1,308 / 12,630 | Val 90.57 %, macro F1 0.882, slow prediction (108.6 s for 12,630 images) |
| **CNN** | **96.01 %** | **504 / 12,630** | Val 98.33 %, macro F1 0.94, weighted F1 0.96 |

### 🔎 Key findings
- Hand-crafted features beat raw pixels by about **6.5 points**: structure matters.
- A CNN beats the SVM by about **6.4 points** and the ANN by about **12.8 points**, with **fewer parameters than the ANN**.
- Shape (HOG) matters far more than colour for this task.

### CNN confusion matrix (test set)
Almost everything sits on the diagonal. The off-diagonal cells show where it gets confused.

![CNN confusion matrix](docs/cnn_confusion_matrix.png)

**Visible confusion patterns**
- **Pedestrians → Road work** (26 images): both are red-bordered warning triangles with small pictograms.
- **Priority road → Yield** (28 images).
- **Speed limit 80 → 50 / 60** (13 each): digits are blurry at 32×32.
- **Right-of-way → Keep right / Dangerous curve right**, and **General caution → Double curve / Traffic signals**.

### CNN: weakest classes
These are mostly **warning triangles with fine inner detail**, several with few training images.


### Use the saved CNN
```python
import keras, numpy as np
from PIL import Image

model = keras.models.load_model("gtsrb_cnn.keras")
img = np.array(Image.open("sign.png").convert("RGB").resize((32, 32)), dtype="float32") / 255.0
probs = model.predict(img[None], verbose=0)[0]
print(np.argmax(probs), probs.max())
```
`traffic_sign_svm.joblib` bundles the scaler, PCA, SVM and class names.

## 📁 Repository Structure
```
├── traffic_sign_recognition.ipynb   # EDA → ANN → CNN → HOG+SVM → comparison
├── gtsrb_cnn.keras                  # trained CNN
├── traffic_sign_svm.joblib          # scaler + PCA + SVM + class names
├── docs/                            # charts used in this README
└── README.md

## 👩‍💻 Connect With Me

**Suhani Saxena **
💡 B.Tech (AI & Data Science)

- 🐙 **GitHub:** [suhanisaxe14-prog](https://github.com/suhanisaxe14-prog)