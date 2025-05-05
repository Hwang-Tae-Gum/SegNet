# SegNet for CamVid Semantic Segmentation

## Description

This project demonstrates a PyTorch implementation of the SegNet model for performing semantic segmentation on the CamVid road scene dataset. SegNet is a deep learning architecture that segments images at the pixel level, predicting the object class (e.g., road, car, pedestrian) for each region.

## Dataset Information (CamVid)

CamVid is a dataset of real-world road scene images with pixel-level object class labels.
Each image has a resolution of 480x360 pixels and is labeled with 32 object classes.
This project utilizes the latest CamVid dataset downloaded via kagglehub and splits it into train/val/test sets.

## Model Architecture (SegNet)

SegNet is a Fully Convolutional Network with an encoder-decoder structure.
The encoder consists of multiple Conv-BatchNorm-ReLU-MaxPool layers that progressively downsample the input image.
The decoder upsamples using indices obtained from the max-pooling layers (MaxUnpool) to restore the original resolution.
The output of the final decoder layer has a number of channels equal to the number of classes, predicting the class with the highest score for each pixel.

## Hyperparameters

* Image Size: 256x256
* Batch Size: 8
* Epochs: 101
* Optimizer: Adam (learning rate=0.001)
* Loss: CrossEntropyLoss
* Device: Automatic selection of CUDA (GPU) or CPU

## Training and Validation Process

The dataset is split into train/val/test sets and loaded using DataLoaders.
In each epoch, the model is trained on the training set and evaluated on the validation set.
The main evaluation metrics are Pixel Accuracy and F1 Score.
Model checkpoints are saved every 5 epochs.
Prediction results are visualized during the training and validation process to assess the model's segmentation performance.

## Improvements and Future Updates

**Current Limitations:**

* Focuses on distinguishing objects from the background within the image rather than detailed per-class performance.
* Limited performance in fine-grained distinctions between objects (classes) and handling complex boundaries.

**Future Directions:**

* Add more diverse evaluation metrics such as per-class IoU (Intersection over Union).
* Apply state-of-the-art techniques such as data augmentation, deeper networks, and attention mechanisms.
* Enhance analysis capabilities with per-class confusion matrices, visualizations, etc.

## Conclusion

This project provides a basic pipeline for semantic segmentation using the SegNet architecture and the CamVid dataset. It can be utilized for pixel-level object recognition in real-world road environments, and further improvements can lead to more precise segmentation performance.
