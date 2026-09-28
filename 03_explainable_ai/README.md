# Explainable AI for Cat and Dog Classification

## Problem Statement

Deep learning models can classify images accurately, but their decisions are often difficult to interpret. Explainable AI techniques help visualize which regions of an image influenced the model's prediction.

## Objective

To develop a Cat vs Dog image classification system using transfer learning with MobileNetV2 and explain its predictions using Grad-CAM.

## Dataset

A Cat and Dog image dataset was used.

Dataset structure:

- Training set: 276 images
- Validation set: 70 images
- Classes:
  - Cat
  - Dog

The dataset contains images of different cat and dog breeds.

The dataset is stored locally and is excluded from GitHub using `.gitignore`.

## Methodology

The project follows these steps:

1. Load the Cat and Dog image dataset.
2. Resize images to 160 × 160 pixels.
3. Apply image augmentation.
4. Use MobileNetV2 pretrained on ImageNet as the feature extractor.
5. Add a classification layer for Cat vs Dog prediction.
6. Train the model using the training dataset.
7. Evaluate the model using the validation dataset.
8. Generate a confusion matrix and classification report.
9. Apply Grad-CAM to visualize important image regions.
10. Overlay the Grad-CAM heatmap on the original image.

## Tools and Libraries

- Python
- TensorFlow
- MobileNetV2
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Results

The model achieved 100% accuracy on the 70-image validation set.

Classification results:

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Cat | 1.00 | 1.00 | 1.00 |
| Dog | 1.00 | 1.00 | 1.00 |

Overall validation accuracy: **1.00**

The result is based on the available 70-image validation set.

## Explainable AI Result

Grad-CAM was applied to the trained Cat vs Dog classifier.

The Grad-CAM visualization highlights image regions that contributed to the model's prediction.

Example prediction:

- Actual class: Cat
- Predicted class: Cat
- Confidence: 99.98%

## Results and Visualizations

- `results/training_accuracy.png`
- `results/training_loss.png`
- `results/confusion_matrix.png`
- `results/classification_report.txt`
- `results/gradcam_cats_dogs.png`

## Conclusion

A MobileNetV2 transfer-learning model was developed for Cat vs Dog classification. The model was evaluated on a validation dataset and achieved 100% validation accuracy on the available 70 images.

Grad-CAM was then used to provide a visual explanation of the model's prediction by highlighting important regions of the input image.

This demonstrates how Explainable AI can make image classification models easier to interpret.
