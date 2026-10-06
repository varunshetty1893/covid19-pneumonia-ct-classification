# LungLens — COVID-19 & Pneumonia CT Scan Classification

Research and educational project that extracts image features with DenseNet121 and compares SVM, Random Forest, and Logistic Regression classifiers.

## Deployment layout

- **Vercel:** static website in `frontend/`
- **Hugging Face Spaces:** Gradio API bundle in `hf-space/`

The Gradio deployment preserves the original TensorFlow DenseNet121 feature extractor and trained scikit-learn models. It runs inference on CPU; Hugging Face ZeroGPU is PyTorch-oriented and will not accelerate this TensorFlow CPU model.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the complete deployment steps. The uploaded source archive did **not** include the trained classifiers. Before the Space can start, train or provide:

```text
backend/artifacts/svm.joblib
backend/artifacts/rf.joblib
backend/artifacts/lr.joblib
backend/artifacts/metrics.json
```

The approximately 3 GB dataset is not included. It is only needed to train the models; do not upload it to Vercel or Hugging Face.

## Train the models

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1ibOUHamULECObSAR1QDvZLw7C_PWZdg8?usp=sharing). Put its class folders under `backend/Mendaly/`:

```text
backend/Mendaly/COVID2_CT/
backend/Mendaly/pneumonia_CT/
backend/Mendaly/Normal_CT/
```

Then, using Python 3.11:

```bash
cd backend
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
```

Training extracts DenseNet121 features and may take a while on CPU. It saves the three classifiers and metrics under `backend/artifacts/`.

## Run locally

After training, from `backend/` run:

```bash
python server.py
```

Open `http://localhost:5173`. For local mode, leave both `window.CT_API` and `window.CT_SPACE` empty in `frontend/config.js`.

## Important

This is a research/educational demo, **not a medical diagnostic tool**. The public Space API has no authentication. Do not upload identifiable or sensitive patient scans.
