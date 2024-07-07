from sklearn.svm import LinearSVC # type: ignore
import numpy as np # type: ignore
import cv2 as cv # type: ignore
import PIL # type: ignore
import os # type: ignore
import joblib # type: ignore

class Model:

    def __init__(self):
        self.model = LinearSVC()
        self.classNames = []

    def train_model(self, photos_dir):
        img_list = np.array([])
        class_list = np.array([])
        counter = 0
        for class_label, class_name in enumerate(os.listdir(photos_dir), start=1):
            class_dir = os.path.join(photos_dir, class_name)
            self.classNames.append(class_name)
            if os.path.isdir(class_dir):
                # Iterate through each image in the current class folder
                for filename in os.listdir(class_dir):
                    if filename.endswith('.jpg') or filename.endswith('.png'):
                        img_path = os.path.join(class_dir, filename)
                        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Read image as grayscale
                        if img is not None:
                            img = img.reshape(307200)
                            img_list = np.append(img_list, [img])
                            class_list = np.append(class_list, class_label)
                            counter += 1

        img_list = img_list.reshape(counter, 307200)
        self.model.fit(img_list, class_list)
        print("Model successfully trained!")

    def predict(self, frame):
        img_path = './SavedModel/photo.jpg'
        cv.imwrite(img_path, frame)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = img.reshape(307200)
        prediction = self.model.predict([img])
        index = int(prediction[0]-1)
        return self.classNames[index]
    
    def save_model(self, model_path='./SavedModel/trained_model.pkl', class_names_path='./SavedModel/class_names.txt'):
        joblib.dump(self.model, model_path)
        with open(class_names_path, 'w') as f:
            for class_name in self.classNames:
                f.write(f"{class_name}\n")
        print("Model and class names saved successfully at ./SavedModel")

    def load_model(self, model_path='trained_model.pkl', class_names_path='class_names.txt'):
        self.model = joblib.load(model_path)
        with open(class_names_path, 'r') as f:
            self.classNames = [line.strip() for line in f]
        print("Model and class names loaded successfully!")