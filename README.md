# PredBrainTumor

A deep learning model for predicting brain tumor from MRI images using TensorFlow Convolutional Neural Network (CNN). Transfer learning is used to train the model. The model has four classes: meningioma, glioma, pituitary tumor, and no tumor with 98% prediction accuracy.

## Description

The model is trained on the [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset from Kaggle. The model is trained on 2,937 images and tested on 327 images. The model has 98% accuracy on the test set.

Sample images from the dataset:
![sample images](https://user-images.githubusercontent.com/13655344/188681990-ebb411b9-356a-4b3f-bd3e-e283092ba15c.jpg)

Epoch vs Training and Validation Accuracy/Loss:
![epoch vs accuracy](https://user-images.githubusercontent.com/13655344/188682950-7d19a10d-c375-487c-a5b1-254df3d4eb9f.jpg)

Model classificaion report:
![classification report](https://user-images.githubusercontent.com/13655344/188683673-26a72394-7198-4ed9-bfbf-a961eb26732f.jpg)

Confusion matrix:
![confusion matrix](https://user-images.githubusercontent.com/13655344/188683945-f82ecc56-2245-40f3-97ac-ee9bbbe5ad3b.png)

_Note: The model is trained on Kaggle's GPU. It takes approximately 5 minutes to train the model for 12 epochs._
