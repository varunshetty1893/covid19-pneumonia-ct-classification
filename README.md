<div align="center">

# 🫁 LungLens

### AI-Powered COVID-19 & Pneumonia CT Scan Classification

Frozen **DenseNet121** features + **SVM**, **Random Forest** and **Logistic Regression** classifiers.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-lunglens--ai.vercel.app-black?logo=vercel)](https://lunglens-ai.vercel.app/)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DenseNet121-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-classifiers-F7931E?logo=scikit-learn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-API-FF7C00)
![Use](https://img.shields.io/badge/Use-Research%20%26%20Education%20only-red)

**[🌐 Try the live demo](https://lunglens-ai.vercel.app/)**

</div>

---

> ⚠️ **Disclaimer:** LungLens is a research and educational project. It is **not a medical diagnostic tool** and must not replace professional medical advice. Do not upload identifiable or sensitive patient scans.

## 📖 Overview

LungLens classifies chest CT images into three classes: **COVID-19**, **Pneumonia** and **Normal**. Rather than training a CNN end to end, it uses transfer learning with classical ML: a pretrained DenseNet121 acts as a fixed feature extractor, and three lightweight classifiers are trained on the extracted features and compared on identical inputs.

## ✨ Features

- **Upload & classify** – drag and drop a JPG/JPEG/PNG chest CT (up to 10 MB) and get a prediction with confidence.
- **Model comparison** – see how SVM, Random Forest and Logistic Regression each classify the same scan.
- **Model Insights page** – performance metrics, confusion matrices, and a side-by-side comparison of the three models.
- **Methodology transparency** – dataset, preprocessing and feature extraction are documented in the app, along with an honest limitations note.

## 🏗️ Pipeline

```
CT Scan → Preprocessing → DenseNet121 (frozen) → 50,176-d feature vector → SVM / RF / LR → Prediction
```

| Stage | Details |
|-------|---------|
| **Dataset** | Three class folders: `COVID2_CT`, `pneumonia_CT`, `Normal_CT`. 70/30 train–test split, stratified by class, `random_state=42`. |
| **Preprocessing** | Read with OpenCV → BGR to RGB → resize to **224×224** → DenseNet `preprocess_input` normalization. |
| **Feature extraction** | DenseNet121 with ImageNet weights, classification head removed, **all layers frozen**. The 7×7×1024 output is flattened into a **50,176-value** vector per image. |
| **Classifiers** | **SVM**, **Random Forest** (ensemble of 100 decision trees), **Logistic Regression** (linear, up to 1,000 iterations). |
| **Metrics** | Weighted metrics on the held-out test set, plus confusion matrices. |

## 📂 Repository structure

```
.
├── backend/                 # Training pipeline + local server
│   ├── train.py             # Extracts features, trains SVM / RF / LR, saves metrics
│   ├── server.py            # Local server (serves frontend + inference)
│   ├── requirements.txt
│   ├── Mendaly/             # Dataset goes here (not included)
│   └── artifacts/           # Generated: svm.joblib, rf.joblib, lr.joblib, metrics.json
├── frontend/                # Static website (Vercel): Home, Classify, Model Insights, About
│   └── config.js            # API / Space endpoint configuration
├── hf-space/                # Gradio API bundle (Hugging Face Spaces)
├── covid_19_&_pneumonia_classification.ipynb   # Original exploration notebook
├── DEPLOYMENT.md            # Full deployment guide
└── README.md
```

## 🗂️ Dataset

The dataset is roughly **3 GB** and is **not included** in this repo. It is only needed for training; do not upload it to Vercel or Hugging Face.

1. Download it from [Google Drive](https://drive.google.com/drive/folders/1ibOUHamULECObSAR1QDvZLw7C_PWZdg8?usp=sharing).
2. Place the class folders under `backend/Mendaly/`:

```
backend/Mendaly/
├── COVID2_CT/
├── pneumonia_CT/
└── Normal_CT/
```

## 🚀 Getting started

**Prerequisites:** Python 3.11, ~3 GB of disk space for the dataset. No GPU needed.

### 1. Train the models

```bash
cd backend
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python train.py
```

Feature extraction can be slow on CPU. When it finishes, `backend/artifacts/` contains `svm.joblib`, `rf.joblib`, `lr.joblib` and `metrics.json`.

> The trained classifiers are not committed to the repo, so run `train.py` (or supply your own artifacts) before starting the app.

### 2. Run locally

```bash
cd backend
python server.py
```

Open **http://localhost:5173**. For local mode, leave both `window.CT_API` and `window.CT_SPACE` empty in `frontend/config.js`.

## ☁️ Deployment

| Component | Platform | Folder |
|-----------|----------|--------|
| Website | [Vercel](https://vercel.com) | `frontend/` |
| Inference API | [Hugging Face Spaces](https://huggingface.co/spaces) (Gradio) | `hf-space/` |

The Space keeps the original TensorFlow DenseNet121 extractor and trained scikit-learn models and runs on **CPU**. Hugging Face ZeroGPU is PyTorch-oriented and will not accelerate this TensorFlow model. The first request after idle can take a few seconds while the model loads.

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for step-by-step instructions.

## 📊 Results

All three models score close to perfect on the test set; the clearest difference is how many test scans each misclassifies. Exact numbers are in `backend/artifacts/metrics.json` and on the app's **Model Insights** page.

| Classifier | Accuracy | Precision | Recall | F1 (weighted) |
|------------|:--------:|:---------:|:------:|:-------------:|
| SVM | – | – | – | – |
| Random Forest | – | – | – | – |
| Logistic Regression | – | – | – | – |

## ⚠️ Limitations

Please read the scores with caution:

- **Possible data leakage.** The split is random by image, so slices from the same patient may appear in both train and test sets. Near-perfect scores can reflect this rather than true generalization.
- **Single split, no cross-validation, one dataset.** There is no external validation.
- **Always predicts one of three classes.** The models will assign a label even to an image that is not a chest CT.
- **Not clinically validated.** This is a research demo only.

## 🔒 Security & privacy

- The public Space API has **no authentication**.
- Never upload identifiable, confidential or real patient data.

## 🔮 Future work

- Patient-level train/test splitting and k-fold cross-validation
- External dataset validation
- Out-of-distribution detection (reject non-CT images)
- Fine-tuning DenseNet121 instead of a frozen extractor
- Grad-CAM explanations
- Authentication and rate limiting for the API

## 🛠️ Tech stack

TensorFlow/Keras (DenseNet121) · scikit-learn · OpenCV · Gradio · Static HTML/JS · Vercel · Hugging Face Spaces

## 🤝 Contributing

Issues and pull requests are welcome. For major changes, please open an issue first.

## 📄 License

No license specified yet. Consider adding one (e.g. [MIT](https://choosealicense.com/licenses/mit/)).

## 🙏 Acknowledgements

- Huang et al., *Densely Connected Convolutional Networks* (DenseNet)
- The open-source communities behind TensorFlow, scikit-learn and Gradio

---

<div align="center">
Built by <a href="https://github.com/varunshetty1893">@varunshetty1893</a>
</div>
