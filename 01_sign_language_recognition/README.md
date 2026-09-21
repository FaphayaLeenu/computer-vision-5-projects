# Vision-Based Sign Language Recognition

## 1. Project Title

Vision-Based Sign Language Recognition Using Convolutional Neural Network

## 2. Problem Statement

Communication can be difficult for people who use sign language when the other person does not understand sign language. A computer vision system can recognize hand signs from images and convert them into corresponding alphabet classes.

## 3. Objective

The objective of this project is to develop a computer vision system that recognizes American Sign Language (ASL) alphabet hand signs from images using a Convolutional Neural Network (CNN).

## 4. Dataset

Dataset: Sign Language MNIST

The dataset contains grayscale images of hand signs.

- Image size: 28 × 28 pixels
- Training images: 27,455
- Test images: 7,172
- Number of classes: 24
- J and Z are excluded because they require movement.

Dataset source:

https://www.kaggle.com/datasets/datamunge/sign-language-mnist

The dataset is stored locally for training and testing but is not uploaded to this GitHub repository.

## 5. Methodology

The project follows these steps:

1. Load the Sign Language MNIST dataset.
2. Separate image pixels and labels.
3. Normalize pixel values from 0–255 to 0–1.
4. Reshape images into 28 × 28 × 1 format.
5. Convert class labels into numerical categories.
6. Build a Convolutional Neural Network.
7. Train the CNN using the training dataset.
8. Evaluate the model using the test dataset.
9. Generate accuracy, precision, recall and F1-score.
10. Generate a confusion matrix to analyze classification performance.

### CNN Architecture

- Convolutional Layer: 32 filters
- Max Pooling Layer
- Convolutional Layer: 64 filters
- Max Pooling Layer
- Flatten Layer
- Dense Layer: 128 neurons
- Dropout Layer
- Output Layer: 24 classes

## 6. Tools and Libraries

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV

## 7. Results

The trained CNN was evaluated on 7,172 test images.

| Metric | Result |
|---|---:|
| Test Accuracy | 93% |
| Macro Precision | 93% |
| Macro Recall | 93% |
| Macro F1-score | 93% |

The model achieved approximately 93% accuracy on the test dataset.

## 8. Performance Evaluation

The following evaluation outputs were generated:

- Training and validation accuracy graph
- Confusion matrix
- Classification report
- Test accuracy
- Precision
- Recall
- F1-score

The confusion matrix shows the correctly and incorrectly classified samples for each sign-language class.

## 9. Project Outputs

Important output files are available in the `results/` folder:

- `sample_sign.png`
- `training_accuracy.png`
- `confusion_matrix.png`
- `classification_report.txt`

The trained model is stored in:

`outputs/sign_language_model.keras`

## 10. Conclusion

A CNN-based vision system was developed to recognize sign-language alphabet hand signs. The model achieved approximately 93% test accuracy on the Sign Language MNIST test dataset. The results demonstrate that CNNs can learn visual features from hand-sign images and classify different sign-language alphabet classes.

## 11. Future Improvements

- Real-time webcam-based sign recognition
- Hand detection and tracking
- Recognition of complete sign-language words
- Recognition of dynamic signs such as J and Z
- Deployment as a real-time application