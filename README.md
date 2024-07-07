
# Object Detection Model with LinearSVC

This project implements an object detection model using the LinearSVC classifier from scikit-learn.
The model is trained on images of different objects and can predict the class (i.e., the name) of a new object image.

# Table of Contents
- [Installation](Installation)
- [Training the Model](Training-the-Model)
- [Saving and Loading the Model](Saving-and-Loading-the-Model)
- [Web Interface](Web-Interface)

# Installation
##### 1. Clone the repository:
```bash
git clone git@github.com:alimourii/ObjectsClassifier.git
```
##### 2. Install the required dependencies:
```bash
pip install numpy opencv-python Pillow scikit-learn joblib Flask
```
# Usage
## Training the Model
##### 1. Organize your training data
if already you have the images of each object, inside photos directory, create a subdirectory for each class (i.e., each object). Place the images of each object in their respective subdirectory. For example:
```
photos/
├── object1/
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── .......
│   ├── .......
│   └── imgN.jpg
│
├── object2/
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── .......
│   ├── .......
│   └── imgN.jpg
│
```
## Using web interface
A simple web interface is provided to interact with the model. The interface allows you to set object names, take photos, train the model, and predict the class of a new image.
