# 🚦 Traffic Sign Recognition: Classical ML (HOG + SVM) vs ANN vs CNN

Traffic sign classification on the **GTSRB** dataset (43 classes). The same data, the same track-wise split and the same test set are used to compare three approaches:

| Approach | How features are obtained |
|---|---|
| **ANN** | raw pixels, flattened |
| **ML: HOG + colour + SVM** | hand-crafted features (shape + colour), classified by an SVM |
| **CNN** | features learned automatically by convolution layers |

## Results (test set, 12,630 images)

| Model | Test accuracy |
|---|---|
| ANN (raw pixels) | 83.17 % |
| ML: HOG + colour + SVM | 89.64 % |
| **CNN** | **96.01 %** |

![Model comparison](images/model_comparison.png)

> Numbers come from one training run with fixed random seeds (42). A re-run can differ slightly, mainly for the ANN and CNN.

### What the experiments show
- **CNN is the best model (96.01 %).** It is about 6.4 points above the SVM and 12.8 points above the ANN, because it learns its own features and keeps the spatial structure of the image.
- **Hand-made features beat a plain ANN.** HOG + SVM is about 6.5 points above the ANN, which sees only flattened raw pixels.
- **HOG carries almost all the information.** Validation accuracy: colour only 13.8 %, HOG only 89.7 %, HOG + colour 90.0 %.
- **RBF kernel beats linear.** Linear SVM 86.3 % vs RBF SVM 90.6 % (validation), so the classes are not linearly separable in HOG space.
- **Weakest classes.** *Pedestrians* and *Beware of ice/snow* are the hardest classes for both models (SVM: 66.7 % and 58.7 %, CNN: 50.0 % and 56.7 %). They are small pictograms that look similar to other warning signs at 32x32 pixels.
- **Cost (CPU only, no GPU).** SVM: about 6 minutes of feature extraction, about 1 minute to train the final model, and about 109 seconds to predict the test set. CNN: about 50 seconds per epoch for 30 epochs.

### Training curves and confusion matrices

| ANN curves | CNN curves |
|---|---|
| ![ANN](images/ann_training_curves.png) | ![CNN](images/cnn_training_curves.png) |

| SVM confusion matrix | CNN confusion matrix |
|---|---|
| ![SVM](images/svm_confusion_matrix.png) | ![CNN](images/cnn_confusion_matrix.png) |

## Method

1. **Data:** GTSRB, 39,209 training images and 12,630 test images, resized to 32x32.
2. **Track-wise split:** each real sign appears in about 30 near-identical frames. A random split would leak the same sign into train and validation and inflate accuracy, so the split is done per track (`GroupShuffleSplit`, 80 / 20: 31,349 train, 7,860 validation). The validation split contains 41 of the 43 classes.
3. **ANN:** Flatten, Dense 256 (ReLU), Dropout 0.3, Dense 128 (ReLU), softmax.
4. **CNN:** data augmentation (active only during training), three Conv blocks (32, 64, 128 filters) with BatchNorm and MaxPooling, Dense 128, Dropout 0.4, softmax. Early stopping on validation loss (patience 5, best weights restored).
5. **ML pipeline:** CLAHE contrast fix, then **HOG** (shape, 1,764 features) and an **HSV colour histogram** (24 features), then StandardScaler (fit on train only), then PCA (95 % variance, 505 features), then **SVM (RBF)**.
6. **Tuning:** C and gamma grid searched on the track-wise validation set (best: C = 10, gamma = scale x 0.5, validation 90.57 %). The test set is used only once, at the end.

## Project structure

```
Traffic-Sign-Recognition/
├── traffic_sign_recognition.ipynb   # EDA, ANN, CNN, HOG + SVM, comparison
├── requirements.txt
├── .gitignore
└── images/                          # result figures used in this README
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook traffic_sign_recognition.ipynb   # Run all cells
```
The dataset is downloaded automatically with `kagglehub`. On a CPU the whole notebook takes roughly 1.5 to 2 hours (SVM tuning and CNN training are the slowest parts).

The notebook also saves `gtsrb_cnn.keras` and `traffic_sign_svm.joblib` next to itself. They are not included in this repository (the SVM file can exceed GitHub's 100 MB limit), so run the notebook to create them.

## Limitations
- Classifies a **cropped sign only**; finding signs in a full road scene needs a detector such as YOLO.
- Trained on **German** signs; Indian signs would need new data.
- 32x32 images lose fine detail (digits on speed limits, small pictograms).
- The CNN's early stopping uses the validation set, so its validation accuracy (98.33 %) is slightly optimistic; the test accuracy is the number to report.

## Future work
- Add a detector (YOLO) for full scenes, and try transfer learning (e.g. MobileNet).
- Try other features (LBP, HOG with different cell sizes) for comparison.
- Deploy the HOG + SVM model as a small web app.

## Dataset
[GTSRB: German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) (Kaggle).
