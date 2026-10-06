# COVID-19 & Pneumonia CT Scan Classification

CT scan image classification of COVID-19, Pneumonia, and Normal cases using **DenseNet121** feature extraction with SVM, Random Forest, and Logistic Regression classifiers. Includes a website (LungLens) to classify an uploaded scan.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Colab](https://img.shields.io/badge/Google-Colab-yellow)

## Pipeline
`CT Scans` → `Preprocessing` → `DenseNet121` → `Feature Vectors` → `ML Classifiers` → `Evaluation`

## Classes
| Class | Folder |
|-------|--------|
| COVID-19 | COVID2_CT |
| Pneumonia | pneumonia_CT |
| Normal | Normal_CT |

## Models Compared
- SVM (Linear kernel)
- Random Forest (100 estimators)
- Logistic Regression (max_iter=1000)

Evaluated on: Accuracy, Precision, Recall, F1 Score

## Dataset
Download from [Google Drive](https://drive.google.com/drive/folders/1ibOUHamULECObSAR1QDvZLw7C_PWZdg8?usp=sharing) (~3GB). Not stored in this repo.

## Notebook (research)
Open `covid_19_&_pneumonia_classification.ipynb` in Google Colab, mount Drive, place the dataset at `/content/drive/MyDrive/DS1/Mendaly/`, run all cells.

## Website
    frontend/   static website (host on Vercel, Root Directory = frontend)
    backend/    Flask API + training script (host on a container service, e.g. Hugging Face Spaces)

### Run locally
    cd backend
    pip install -r requirements.txt
    python train.py        # once; looks for ./Mendaly or ../Mendaly, saves models to backend/artifacts
    python server.py       # open http://localhost:5173

### Deploy
1. Backend: push `backend/` (with `artifacts/`) to a Docker Hugging Face Space (`git lfs track "*.joblib"`). Check `<space-url>/health`.
2. Frontend: set the Space URL in `frontend/config.js` (`window.CT_API = "https://..."`).
3. Vercel: import this repo, set **Root Directory = frontend**, Framework Preset = Other, Deploy.

## Disclaimer
Research and educational use only. Not a medical diagnostic tool.
