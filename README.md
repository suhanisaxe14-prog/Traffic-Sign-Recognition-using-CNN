Traffic Sign Recognition using CNN (GTSRB)
An image classification model that looks at a photo of a road sign and tells which of 43 traffic sign types it is (Stop, Yield, Speed limit 50, No entry, and so on).

1. What problem does this project solve?
Road signs tell drivers what to do: slow down, stop, give way, not enter. A driver-assistance or self-driving system has to read these signs automatically. Doing this by hand-written rules is very hard, because the same sign looks different in different light, angles, distances and weather.

This project teaches a computer to recognise signs from examples. It is shown thousands of labelled sign images, and it learns the patterns (shape, colour, symbol) by itself.

2. What does the model do?
Input	A colour image of one traffic sign (resized to 32x32 pixels)
Output	The sign class (one of 43) and the model's confidence for each class
Example	Image of a red circle with "50" -> Speed limit (50 km/h), 96% confidence
The model is a Convolutional Neural Network (CNN). Early layers detect simple things (edges, corners, colours); deeper layers combine them into shapes and symbols (a triangle, an arrow, a number). The last layer gives a probability for each of the 43 classes, and the highest one is the prediction.

3. Where can it be used?
A model like this is the sign-recognition stage of larger systems:

Driver-assistance systems (ADAS): show the current speed limit on the dashboard, warn the driver about a Stop or No Entry sign.
Self-driving and autonomous vehicles: read road rules from the environment.
Road-maintenance and mapping: automatically scan road videos to list which signs exist and spot missing or damaged ones.
Learning and research: a standard benchmark for studying image classification.
Important: this project is only the classification step. In a real vehicle, a separate detection model (for example YOLO) must first find where the sign is in the camera frame, crop it, and then a classifier like this one reads it.

4. Dataset
GTSRB (German Traffic Sign Recognition Benchmark)

39,209 training images and 12,630 test images
43 classes
Real-world photos with different lighting, blur and angles
Chosen because it is a large, labelled, widely used benchmark, which lets results be compared with other work
5. How the project works
Load data and resize every image to 32x32.
Explore data: sample images, class distribution (some classes have far fewer images).
Normalise pixel values to the 0-1 range.
Split into train and validation without leakage. Each real sign appears in about 30 almost identical frames (a "track"). A random split would put frames of the same sign into both train and validation, so the model would be tested on signs it had already seen and the accuracy would look falsely high. The split here is done track-wise (GroupShuffleSplit), so validation contains only unseen signs.
ANN baseline: a simple fully connected network, used for comparison.
CNN with data augmentation: random rotation, zoom and shift make the model robust to real-world changes. Flips are not used, because they would turn a "left" sign into a "right" sign.
EarlyStopping: training stops when validation loss stops improving, and the best weights are restored.
Evaluate on the untouched test set: classification report, confusion matrix, weakest classes, wrong predictions.
CNN architecture
Input (32x32x3)
-> Data augmentation (rotation, zoom, translation)
-> Conv2D(32)  + BatchNorm + MaxPool
-> Conv2D(64)  + BatchNorm + MaxPool
-> Conv2D(128) + BatchNorm + MaxPool
-> Flatten -> Dense(128) + Dropout(0.4)
-> Dense(43, softmax)
About 362K parameters. Optimizer: Adam (learning rate 0.001). Loss: sparse categorical cross-entropy. Batch size 64.

6. Results
Model	Test accuracy
ANN (baseline)	82.16%
CNN	92.33%
CNN validation accuracy was 93.41%, close to the test accuracy, and the training and validation curves stay near each other, so the model is not overfitting.
The CNN is about 10 percentage points better than the ANN, because convolution layers use the spatial structure of images while an ANN flattens it away.
969 of 12,630 test images were misclassified.
Weakest classes: Pedestrians (51.7%), Beware of ice/snow (54.0%), Double curve (67.8%), Speed limit 60 km/h (75.8%), Speed limit 80 km/h (81.0%). These are similar-looking triangular warning signs, or signs that differ only by a number that is hard to read at 32x32 pixels, and some have fewer training images.

7. Limitations
Classifies a cropped sign only; it cannot find signs in a full road scene.
Trained on German signs. Indian signs differ in some designs and text, so the model would need to be retrained or fine-tuned on Indian sign data before use in India.
32x32 images lose detail; night, heavy rain, glare or damaged signs can reduce accuracy.
Validation contains 41 of 43 classes (two rare classes fell entirely into training). The test set contains all 43.
The duplicate check finds exact duplicates only, not near-duplicate frames of the same track.
8. Future work
Add a sign detection stage (YOLO) to make a full pipeline
Collect or use an Indian traffic sign dataset
Larger image size (48x48 or 64x64) and class weights for rare classes
Grad-CAM heatmaps to show which part of the sign the model looks at
Transfer learning (for example MobileNetV2) for comparison
Streamlit app: upload a sign photo and see the top-3 predictions
9. How to run
Open DL_project.ipynb in Google Colab (GPU recommended).
Run all cells from top to bottom. The dataset downloads automatically through kagglehub.
The first run loads and caches the images in about a minute; training takes roughly 15-25 minutes.
Requirements: Python 3, TensorFlow/Keras 3, NumPy, pandas, scikit-learn, Matplotlib, Pillow, kagglehub.

Author
Suhani Saxena (@suhanisaxe14-prog)
