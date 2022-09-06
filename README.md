# PredBrainTumor

A deep learning model for predicting brain tumor from MRI images using TensorFlow Convolutional Neural Network (CNN). Transfer learning is used to train the model. The model has four classes: meningioma, glioma, pituitary tumor, and no tumor with 98% prediction accuracy.

The model is deployed on here: <https://btpred.soleyman.xyz>

## Built With

- [TensorFlow](https://www.tensorflow.org/) - The machine learning framework used
- [Keras](https://keras.io/) - The high-level neural networks API used
- [EfficientNet](https://keras.io/api/applications/efficientnet/) - For transfer learning
- [Flask](https://flask.palletsprojects.com/) - The web framework used
- [OpenCV](https://opencv.org/) - For image processing
- [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) - For data visualization and analysis

## Run Locally

- Clone the project ```git clone https://github.com/IMSoley/PredBrainTumor```
- Install dependencies ```pip install -r requirements.txt```
- Create upload dir ```mkdir static/uploads```
- Run the app ```python app.py```
- Visit ```http://127.0.0.1:5000```

The [```model```](/model/) folder contains the trained model. Model testing data is in the [```testing```](/static/testing/) folder. These data can be used to test the model locally or on the deployed [website](https://btpred.soleyman.xyz/).

## Description

The model is trained on the [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset from Kaggle. The model is trained on 2,937 images and tested on 327 images. The model has 98% accuracy on the test set.

### Sample images from the dataset

![sample images](https://user-images.githubusercontent.com/13655344/188681990-ebb411b9-356a-4b3f-bd3e-e283092ba15c.jpg)

### Model is defined as follows

```python
    # EfficientNetB0 a convolutional neural network that is trained on more than a million images from the ImageNet database
    effnet = EfficientNetB0(weights='imagenet', # using the weights from the ImageNet database
                        include_top=False,
                        input_shape=(image_size, image_size, 3))
    # effnet output is the input to the model
    model = effnet.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dropout(rate=0.5)(model)
    # this is the output layer with 4 classes
    model = tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax)(model)
    model = tf.keras.models.Model(inputs=effnet.input, outputs=model)
```

### Epoch vs Training and Validation Accuracy/Loss

![epoch vs accuracy](https://user-images.githubusercontent.com/13655344/188682950-7d19a10d-c375-487c-a5b1-254df3d4eb9f.jpg)

### Model classificaion report

```python
    # calling the classification report function
    get_classification_report()
```

![classification report](https://user-images.githubusercontent.com/13655344/188683673-26a72394-7198-4ed9-bfbf-a961eb26732f.jpg)

### Confusion matrix

```python
    # calling the confusion matrix function
    cm_analysis(y_test_new, prediction, "fig", classes, labels)
```

![confusion matrix](https://user-images.githubusercontent.com/13655344/188683945-f82ecc56-2245-40f3-97ac-ee9bbbe5ad3b.png)

_Note: The model is trained on Kaggle's GPU. It takes approximately 5 minutes to train the model for 12 epochs._

## Deployment

### Home page of the deployed model's app

![deployed model](https://user-images.githubusercontent.com/13655344/188695866-c0ad4606-a3e5-45f6-b588-1829e808e13f.png)

### Prediction for each class

| glioma_tumor(0)        | no_tumor(1)           | meningioma_tumor(2)  | pituitary_tumor(3) |
| :-----------: |:-------------:| :----:| :--:|
| <img src="https://user-images.githubusercontent.com/13655344/188699812-e37f496e-c522-456c-98ff-56f73e84cf56.png" width="170">      | <img src="https://user-images.githubusercontent.com/13655344/188699930-bd84bea7-68b3-483a-82fc-c78d60ec27b4.png" width="170"> | <img src="https://user-images.githubusercontent.com/13655344/188700003-a96e1438-3fa2-4244-9f91-7167de3d9b12.png" width="170"> | <img src="https://user-images.githubusercontent.com/13655344/188700094-05207f01-39ac-44c2-86b3-e35cecc24158.png" width="170"> |

## License

MIT License
