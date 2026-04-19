# 🐱🐶 Cat, Dog & Neither Classification

A deep learning web app that classifies images as **cat**, **dog**, or **neither** — powered by MobileNetV2 transfer learning and deployed with Flask.

---

## 🚀 Demo

Upload any image → get an instant prediction with confidence score.

---

## 🧠 How It Works

1. User uploads an image via the Flask web interface
2. Image is preprocessed and passed to a fine-tuned MobileNetV2 model
3. Model predicts one of three classes: `Cat`, `Dog`, or `Neither`
4. Result is displayed in real time

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | MobileNetV2 (Transfer Learning) |
| Framework | TensorFlow / Keras |
| Backend | Flask |
| Language | Python |
| Augmentation | RandomFlip, RandomRotation, RandomZoom |
| Optimization | EarlyStopping, ReduceLROnPlateau |

---

## 📁 Project Structure

```
CatnDog_Classification/
│
├── static/
│   └── uploads/          # Uploaded images
├── templates/
│   └── index.html        # Web UI
├── model/
│   └── model.h5          # Trained model
├── app.py                # Flask app
├── train.py              # Model training script
└── requirements.txt
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Gargie-ui/CatnDog_Classification
cd CatnDog_Classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## 🏋️ Model Training

- Base model: **MobileNetV2** (pretrained on ImageNet, top layers removed)
- Custom layers: `GlobalAveragePooling2D` → `Dense` → `Dropout` → `Dense(3, softmax)`
- Data augmentation applied to reduce overfitting
- Callbacks: `EarlyStopping` and `ReduceLROnPlateau` for optimized training

---

## 📦 Requirements

```
tensorflow
flask
numpy
Pillow
```

---

## 👩‍💻 Author

**Gargi Channe**
- 🔗 [LinkedIn](https://www.linkedin.com/in/gargi-channe)
- 🐙 [GitHub](https://github.com/Gargie-ui)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
