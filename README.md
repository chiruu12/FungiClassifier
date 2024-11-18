# Mushroom Classifier

![Streamlit](https://img.shields.io/badge/streamlit-v1.36.0-brightgreen)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.8.0-orange)

## Project Overview

This project implements a machine learning model to classify mushrooms based on images. The dataset consists of 50 mushroom species, 25 edible and 25 poisonous. The goal is to identify mushrooms and provide accurate safety information.

## Dataset

The dataset used in this project is sourced from Kaggle and contains images of 50 mushroom species (25 edible and 25 poisonous). The dataset can be found on Kaggle here: [Mushroom Classification Dataset](https://www.kaggle.com/datasets/yoonjunggyu/25-edible-mushroom-and-25-poisonous-mushroom/data).

### Dataset Breakdown

| Category           | Number of Species | Total Images |
| ------------------ | ----------------- | ------------ |
| **Edible Mushrooms** | 25                | 1400 images     |
| **Poisonous Mushrooms** | 25             | 1420 images     |

**Total Images**:  2820

We used an 80-20 split for training and validation. The dataset includes images of mushrooms across various angles and lighting conditions.

## Model

The model architecture is based on Convolutional Neural Networks (CNNs). It uses TensorFlow and Keras for training and inference. The model is designed to achieve high accuracy and performance in classifying mushroom species.

### Key Features:
- **Multi-class Classification**: The model can classify between 50 different mushroom species.
- **Accuracy**: Achieved an accuracy of 89% on the test set.
- **Real-time Classification**: The model is integrated with a Streamlit app for easy interaction.


## Convolutional Neural Network (CNN)

A CNN is a type of deep learning algorithm specifically designed to process and analyze visual data, making it ideal for image classification tasks. In this project, the CNN model is trained to recognize different mushroom species based on the images provided in the dataset.

For more information on CNNs, refer to this [beginner’s guide](https://www.analyticsvidhya.com/blog/2021/06/image-processing-using-cnn-a-beginners-guide/).

## Model Architecture
The model used in this project is based on ResNet50, a powerful pre-trained Convolutional Neural Network (CNN) that has been fine-tuned for the task of classifying mushroom species. ResNet150 is widely used for image classification tasks due to its deep architecture and residual connections that help prevent vanishing gradient problems.

### Key Features:

- **ResNet150 Architecture**: The model uses the ResNet150 architecture, which is designed with deep residual networks to improve training performance and prevent overfitting by using skip connections, making it well-suited for complex image recognition tasks like mushroom classification.

- **Fine-tuning**: The ResNet150 model is fine-tuned on our specific mushroom dataset to enhance its ability to distinguish between edible and poisonous species.

- **Transfer Learning**: We use transfer learning by leveraging pre-trained ImageNet weights, enabling the model to recognize important features like textures and edges, which enhances classification accuracy with fewer training samples.

## Web App Integration

The mushroom classification model has been integrated into a web application using **Gemini**. Gemini is a powerful framework that allows us to seamlessly integrate machine learning models into interactive web platforms. The web app provides users with a user-friendly interface for uploading mushroom images and receiving real-time predictions from the trained model. This integration ensures a smooth and efficient user experience, providing access to mushroom classification directly through a browser interface.

## Data Augmentation
  
  Data augmentation techniques are applied to improve model generalization:  
- **Random Flip**: Flips images horizontally.
- **Random Rotation**: Rotates images randomly.
- **Random Zoom**: Zooms in or out on the images.


These augmentations help simulate variations in the data, leading to better model robustness.


## Model Saving and Checkpoints

The model weights are saved using the `ModelCheckpoint` callback during training. This ensures that the best performing model (based on validation accuracy) is stored for later use.
```python
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)
```

## Source

For more information on image classification with TensorFlow, refer to the official [TensorFlow image classification tutorial](https://www.tensorflow.org/tutorials/images/classification).

## Caution:

Please note that the notebooks provided in this repository may not exactly match the final model used for deployment. Modifications and optimizations were made after the notebook experiments to improve model performance, ensure better accuracy, and adapt the model for production use. Therefore, the code in the notebooks should be considered a reference and may differ from the final working solution.

