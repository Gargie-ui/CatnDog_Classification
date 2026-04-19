# 🐱🐶 Cat, Dog & Neither Classification

A deep learning web app that classifies images as **cat**, **dog**, or **neither** — powered by MobileNetV2 transfer learning and deployed with Flask.

## Demo

**[Try it live](https://gigishot-cat-dog-neither-classification.hf.space/)**
Upload any image → get an instant prediction with confidence score.

---

## How It Works

1. User uploads an image via the Flask web interface
2. Image is preprocessed and passed to a fine-tuned MobileNetV2 model
3. Model predicts one of three classes: `Cat`, `Dog`, or `Neither`
4. Result is displayed in real time

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | MobileNetV2 (Transfer Learning) |
| Framework | TensorFlow / Keras |
| Backend | Flask |
| Language | Python |
| Augmentation | RandomFlip, RandomRotation, RandomZoom |
| Optimization | EarlyStopping, ReduceLROnPlateau |

---

## Model Training

- Base model: **MobileNetV2** (pretrained on ImageNet, top layers removed)
- Custom layers: `GlobalAveragePooling2D` → `Dense` → `Dropout` → `Dense(3, softmax)`
- Data augmentation applied to reduce overfitting
- Callbacks: `EarlyStopping` and `ReduceLROnPlateau` for optimized training

---

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
