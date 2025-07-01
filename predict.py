import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
CLASSES = ['invoice', 'note', 'sign']

# Load the trained model once
model = tf.keras.models.load_model('mobilenetv2_text_classifier.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    pred = model.predict(img)
    class_idx = np.argmax(pred, axis=1)[0]
    return CLASSES[class_idx]