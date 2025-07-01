# Image-Based Text Classifier

A machine learning pipeline built for classifying image-based handwritten or printed text into categories like Invoice, Note, and Sign. Developed using TensorFlow and deployed with a Flask web interface.

---

## 📂 Project Structure

```
text-classifier/
├── main.py                    # Flask app server
├── predict.py                 # Model loading + prediction
├── train.py                   # Training script with augmentation
├── requirements.txt           # Python dependencies
├── dataset/                   # Labeled image folders
    ├── invoice/...
    ├── note/...
    └── sign/...
├── templates/       
    └── index.html
└── mobilenetv2_text_classifier.h5  # Trained model (legacy format)
```

---

## 🧪 Dataset

- **Classes**: `invoice`, `note`, `sign`
- **Images per class**: \~30 (manually curated + public datasets)
- **Sources**:
  - [IAM Handwriting Forms](https://www.kaggle.com/datasets/naderabdalghani/iam-handwritten-forms-dataset)
  - [SROIE Invoices](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2)
  - [GTSRB Road Signs](https://www.kaggle.com/datasets/daniildeltsov/traffic-signs-gtsrb-plus-162-custom-classes)

> ⚠️ The `sign` class includes only **symbolic road signs**, not textual signage.

---

## 🧼 Preprocessing & Augmentation

- Resize to `224x224`
- RGB channels retained
- Pixel normalization to `[0, 1]`
- Augmentation using `ImageDataGenerator`:
  - Rotation ±10°
  - Shear, zoom
  - Brightness shift
  - Width/height shifts

---

## 🧠 Model Architecture

- **Base**: MobileNetV2 (frozen weights from ImageNet)
- **Head**:

```text
Input → MobileNetV2 → GlobalAveragePooling → Dropout(0.2) → Dense(3, softmax)
```

- **Training Configuration:**
  - Optimizer: Adam (lr = 0.0005)
  - Loss: Categorical Crossentropy
  - Epochs: 10
  - Batch Size: 16
  
---

## 📊 Training Metrics

### Training Accuracy vs Validation Accuracy

```
| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|:-----:|:------------------:|:-------------------:|:--------------:|:----------------:|
| 1     | 0.2967             | 0.5556              | 1.3772         | 0.9995           |
| 2     | 0.5272             | 0.6111              | 0.9873         | 0.7125           |
| 3     | 0.7994             | 0.8889              | 0.6430         | 0.5220           |
| 4     | 0.8069             | 1.0000              | 0.5639         | 0.3846           |
| 5     | 0.9152             | 1.0000              | 0.4033         | 0.2883           |
| 6     | 0.9607             | 1.0000              | 0.3007         | 0.2202           |
| 7     | 0.9793             | 1.0000              | 0.2645         | 0.1729           |
| 8     | 1.0000             | 1.0000              | 0.2265         | 0.1394           |
| 9     | 1.0000             | 1.0000              | 0.1772         | 0.1153           |
| 10    | 0.9840             | 1.0000              | 0.1403         | 0.0975           |
```

### Confusion Matrix

```
Confusion Matrix:
Labels: ['invoice', 'note', 'sign']
[[6 0 0]
 [0 6 0]
 [0 0 6]]
```


> All classes achieved **100% accuracy** on the validation set:

```
              precision    recall  f1-score   support
     invoice       1.00      1.00      1.00         6
        note       1.00      1.00      1.00         6
        sign       1.00      1.00      1.00         6
```

---

## 🚀 How to Run This Project

### ✅ 1. Clone the Repo

```bash
git clone https://github.com/yourusername/text-classifier.git
cd text-classifier
```

### ✅ 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
```

### ✅ 3. Install Requirements

```bash
pip install -r requirements.txt
```

### ✅ 4. Run Training (Optional)

```bash
python train.py
```

### ✅ 5. Launch Flask App

```bash
python main.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### ✅ 6. Predict via API (Optional)

```bash
curl -X POST -F image=@path_to_image.jpg http://127.0.0.1:5000/api/predict
```

---

## ✅ Challenges Faced

- Defining "sign" class meaning clearly — resolved with symbolic examples
- Preventing overfitting on a small dataset — solved via augmentation + dropout
- Limited image data required manual curation and synthetic generation

---

## 🔮 Future Improvements

- Switch to `model.keras` format
- Add OCR-enhanced classification pipeline
- Expand dataset to include `List`, `Form`
- Host on Render/Hugging Face Spaces for live inference

---

## ✉️ Submission Info

- GitHub Repo: [https://github.com/MinoshPerera/text-classification-project](https://github.com/MinoshPerera/text-classification-project)
- Contact: sithijaperera3@gmail.com
- Submitted on: July 1, 2025

---


