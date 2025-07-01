import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
DATASET_PATH = 'dataset/'
CLASSES = ['invoice', 'note', 'sign']

def load_images():
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATASET_PATH, cls)
        for file in os.listdir(folder):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # or IMREAD_COLOR
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype('float32') / 255.0
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

# Load & Split
X, y = load_images()
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # for grayscale CNN input

# Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2,
     stratify=y, random_state=42
)
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)