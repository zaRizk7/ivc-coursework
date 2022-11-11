# Image and Vision Computing Mini-Project

## Abstract

This describes the Image and Vision Computing assessed practical. The main goal is to compare sensitivities of a classical and a deep learning method as applied to image classification. The core issue here is the robustness of the classification. There are 3 stages to this assignment, and you must do all 3 for full marks. You must do this project with a partner.

## Task Background

The dataset consists of 3035 images belonging to one of 8 classes. The classes are as follows: Black
Widow, Captain America, Doctor Strange, Hulk, Ironman, Loki, Spiderman, and Thanos. The dataset
is already split into training and validation folders for you. You will need to train two models, one
“classical” and one based on “deep learning” to classify these images. Try to make both models as small
as possible while keeping accuracy as high as possible. There is no expected F1 score and mostly your
approach to the problem will be assessed.

The assignment is to:

1. Implement 2 classifiers, one based on “deep learning”, and the other based on a classical method. (You are allowed to use pre-trained model weights for this).
2. Train both classifiers to perform multi-class classification on the dataset and calculate the F1-score for each using the validation data provided. (You may use image augmentation techniques as you see fit to increase your F1 scores).
3. Compare the robustness of the classifiers as progressively stronger perturbations are applied to the test images.
4. Report the results of evaluation of robustness.

## Resources

The data for this can be found at: https://www.kaggle.com/datasets/hchen13/marvel-heroes

A link to a resized dataset will be available in Learn soon. In the meantime, try writing your own code to resize all the images. 

## Classifier Details

A possible classical method can be based on the HoG and/or SIFT feature extraction combined with an SVM classifier. The “deep learning” method can be a CNN such as ResNet.

The work below can be done using either Matlab or Python (or any other language that you are familiar with). You are expected to find implementations of the key technical components from public resources but connect the components together yourself for doing the robustness exploration. You are not expected to code up the algorithms yourself. Below you will find a starting point for 2 possible classifiers that you could use for the project: The ResNet-18 model and the SVM. However, you are free to explore any other classical or deep learning approach to classifying these images.

## ResNet-18

The ResNet18 classifier is an end-to-end classifier that has an image as input and, in this case, has 8 outputs. You will use a ‘pretrained’ network but will need to do some additional re-training so it classifies just the 8 characters. Ideally, this means ‘freezing’ the convolutional layers, replacing the fully connected layers for 8 classes, and then retraining just the fully connected layers. However, don’t worry if you cannot ‘freeze’ the convolutional layers, and have to tune the whole pre-trained network. You can do data augmentation to improve the training. This should be limited to translation, rotation, flipping, and scale variations. You might explore some of the values of the hyperparameters associated with the fully connected layers, such as the number of layers, size of layers, learning rate, etc. You should split the training data into training and validation, but not touch the test data until the inference stage.
There are several sources for pretrained ResNet18 in either Matlab or Python: 
- MATLAB: https://www.mathworks.com/help/deeplearning/ref/resnet18.html
- PyTorch: https://pytorch.org/docs/stable/torchvision/models.html or pytorch: https://docs.fast.ai/quickstart.html

You are welcome to use other sources. 

## SVM

For the input feature to the SVM. Here we use HoG and/or SIFT feature detection in OpenCV and from this create a Bag of Visual Words (BoVW) for each image which we then use to train the SVM model. Python code for HOG can be seen in the link below, similar MATLAB code can also be found: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html

Some Matlab SVM code can be found in the following links listed below: https://uk.mathworks.com/help/stats/support-vector-machinesforbinaryclassification.html or https://uk.mathworks.com/discovery/support-vectormachine.html
Some Python SVM code is at: https://scikit-learn.org/stable/modules/svm.html

As with the ResNet18 training, one should use the training and validation data samples to tune the algorithm, before computing test performance with the independent test data subset.

## Robustness Exploration

Once the two image classification approaches are trained and working well, there are a set of robustness evaluation experiments that need to be run using the trained classifiers. The central idea is that a variety of perturbations will be applied to the images, and then the classification rate will be re-calculated for both of the classifiers. You will produce a plot of classification accuracy versus amount of perturbation for each of the perturbations. Don’t retrain the classifiers, just evaluate their performance on the perturbed data. The evaluation metric is mean classification accuracy on the test set. Evaluation will use increasing levels of perturbation applied to the test set. The plots will show the change in accuracy with respect to the amount of perturbation. In total, there are 8 perturbations resulting in 8 plots. Show examples of each perturbation in your report.

## Gaussian Pixel Noise

To each pixel, add a Gaussian distributed random number with 10 increasing standard deviations from {0, 2, 4, 6, 8, 10, 12, 14, 16, 18 }. Make sure that the pixel values are integers in the range 0..255 (e.g. replace negative numbers by 0, values > 255 by 255). 

## Gaussian Blur

Create test images by blurring the original image by 3x3 mask:

$$
\frac{1}{16}\left(\begin{array}{cc} 
1 & 2 & 1\\
2 & 4 & 2\\
1 & 2 & 1
\end{array}\right)
$$
