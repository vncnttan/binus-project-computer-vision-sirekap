import os
import cv2
from keras.applications.resnet import preprocess_input as resnet_preprocess_input
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from functools import cmp_to_key
from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
import numpy as np

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ZEWEc1ZPB3XNtTHuMvf2"
)

MODEL_ID = "kertas-suara-titik/1"
IMAGE_SIZE = (40, 240)

def compare(a, b):
    return 1 if (a['x'] + a['y']) > (b['x'] + b['y']) else -1

def cropping(path):
    img = cv2.imread(path)
    image_path = path
    result = CLIENT.infer(image_path, model_id=MODEL_ID)

    if len(result['predictions']) != 9:
        print("Please provide a better photo..")
        return []

    pred = result['predictions']
    pred = sorted(pred, key=cmp_to_key(compare))
    cropped_imgs = []
    for res in pred:
        cropped_img = img[int(res['y'] - (res['height'] / 2)): int(res['y'] + (res['height'] / 2)),
                          int(res['x'] - (res['width'] / 2)): int(res['x'] + (res['width'] / 2))]
        cropped_img = cv2.resize(cropped_img, IMAGE_SIZE)
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        th = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
        th = th / 255.0
        cropped_imgs.append(th)

    return cropped_imgs

def predict(cropped_imgs, model_file, traditional):
    if traditional:
        with open(f'models/{model_file}', 'rb') as file:
            model = pickle.load(file)
    else:
        model = load_model(f'models/{model_file}')
    
    tmp_images = np.array(cropped_imgs)

    if traditional:
        tmp_images = tmp_images.reshape(tmp_images.shape[0], -1)
    else:
        tmp_images = np.repeat(tmp_images[..., np.newaxis], 3, -1)
        if model_file == 'ResNet50PreprocessInput.h5':
            tmp_images = resnet_preprocess_input(tmp_images)
        elif model_file == 'VGG19.h5':
            tmp_images = vgg19_preprocess_input(tmp_images)

    if traditional:
        return model.predict(tmp_images)

    y_pred = model.predict(tmp_images)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred

def show_output(cropped_imgs, predictions):
    plt.figure(figsize=(2, 6))
    labels = ["Paslon 1", "Paslon 2", "Paslon 3"]
    for i, label in enumerate(labels):
        plt.subplot(6, 3, 2 + (6 * i))
        plt.axis("off")
        plt.text(0.5, 0.5, label, fontsize=12, ha='center', va='center')

    for i, (cImg, cPred) in enumerate(zip(cropped_imgs, predictions)):
        mul = (i // 3) + 1
        plt.subplot(6, 3, (mul * 3) + i + 1)
        plt.axis("off")
        fig = plt.imshow(cImg, cmap='gray')
        plt.title(cPred)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    path = input("Enter Image Path: ")
    models = [['ResNet50PreprocessInput.h5', False], ['VGG19.h5', False], ['RandomForest.pkl', True], ['SVM.pkl', True]]
    while True:
        print("Select the model you want to use for prediction: ")
        print("1. Resnet50")
        print("2. VGG19")
        print("3. Random Forest")
        print("4. Support Vector Machine")
        print(">> ", end="")
        try:
            model_input = int(input())
            if 1 <= model_input <= 4:
                break
            else:
                print("Have to be between 1 - 4")
        except ValueError:
            print("Invalid input, please enter a number between 1 and 4.")

    cropped_imgs = cropping(path)
    if cropped_imgs:
        output_pred = predict(cropped_imgs, models[model_input - 1][0], models[model_input - 1][1])
        show_output(cropped_imgs, output_pred)


