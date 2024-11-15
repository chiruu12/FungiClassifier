# Mushroom Classification

This is an image classification project aimed at detecting various mushroom species using **Keras** and **TensorFlow**. The project leverages Convolutional Neural Networks (CNNs) to classify images of mushrooms into different species. The dataset used consists of images from 9 different mushroom species, and the model is trained to distinguish between them based on their visual characteristics.

**Libraries Used**:
- TensorFlow: [TensorFlow Documentation](https://www.tensorflow.org/)
- Keras: [Keras Documentation](https://keras.io/)

## Dataset

The dataset used in this project is sourced from Kaggle and contains mushroom images organized by species. The species are as follows:

| Mushroom Species | Number of Images |
| ----------------- | ---------------- |
| **Agaricus**      | 353              |
| **Amanita**       | 750              |
| **Boletus**       | 1073             |
| **Cortinarius**   | 836              |
| **Entoloma**      | 364              |
| **Hygrocybe**     | 316              |
| **Lactarius**     | 1563             |
| **Russula**       | 1148             |
| **Suillus**       | 311              |

The dataset can be found on Kaggle here: [Mushroom Classification Dataset](https://www.kaggle.com/datasets/lizhecheng/mushroom-classification/data)

**Total Images**:  
- **Training Set**: 5,372 images  
- **Validation Set**: 1,342 images  
- **Total**: 6,714 images  

We used an 80-20 split for training and validation.

### Directory Structure

Each species is placed into its respective subfolder, allowing for easy loading of the dataset using TensorFlow’s `image_dataset_from_directory` method, which labels the data according to the folder names(The images names are not same as shown below).
```
  Mushrooms/  
  ├── Agaricus/  
  │ ├── 001.jpg  
  │ ├── 002.jpg  
  │ └── ...  
  ├── Amanita/  
  │ ├── 001.jpg  
  │ ├── 002.jpg  
  │ └── ...  
  ├── Boletus/  
  │ ├── 001.jpg  
  │ ├── 002.jpg  
  │ └── ...  
  ├── Cortinarius/  
  │ ├── 001.jpg  
  │ ├── 002.jpg  
  │ └── ...  
  ├── Entoloma/  
  │ ├── 001.jpg  
  │ ├── 002.jpg  
  │ └── ...  
  ├── Hygrocybe/  
  │ ├── 001.jpg  
  │ ├── 002.jpg  
  │ └── ...  
  ├── Lactarius/  
  │ ├── 001.jpg  
  │ ├── 002.jpg  
  │ └── ...  
  ├── Russula/  
  │ ├── 001.jpg  
  │ ├── 002.jpg  
  │ └── ...  
  └── Suillus/  
  ├── 001.jpg  
  ├── 002.jpg  
  └── ...
  ```


## Convolutional Neural Network (CNN)

A CNN is a type of deep learning algorithm specifically designed to process and analyze visual data, making it ideal for image classification tasks. In this project, the CNN model is trained to recognize different mushroom species based on the images provided in the dataset.

For more information on CNNs, refer to this [beginner’s guide](https://www.analyticsvidhya.com/blog/2021/06/image-processing-using-cnn-a-beginners-guide/).

## Model Architecture
The model used in this project is based on ResNet50, a powerful pre-trained Convolutional Neural Network (CNN) that has been fine-tuned for the task of classifying mushroom species. ResNet50 is widely used for image classification tasks due to its deep architecture and residual connections that help prevent vanishing gradient problems.

### Key Features:

- **Pre-trained on ImageNet** : The ResNet50 model is pre-trained on the ImageNet dataset, which allows it to learn general features (such as edges, textures, etc.) that are useful for various image recognition tasks.

- **Fine-tuning** : The model is further fine-tuned to recognize mushroom species from our specific dataset by training the top layers on our mushroom images.

- **Transfer Learning** : Using transfer learning, we leverage the knowledge gained from ImageNet to improve the accuracy of our mushroom classification model with fewer training samples.



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

