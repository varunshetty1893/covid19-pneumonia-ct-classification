---
title: LungLens CT Classifier
emoji: 🩻
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
python_version: "3.10"
pinned: false
---

# LungLens CT Classifier

Research and educational demo for classifying CT images. The Space expects these files in `artifacts/`:

- `svm.joblib`
- `rf.joblib`
- `lr.joblib`
- `metrics.json`

The app runs the existing TensorFlow DenseNet121 feature extractor and scikit-learn classifiers on CPU. ZeroGPU is not used by this TensorFlow CPU model.

**Not a medical diagnostic tool. Do not upload identifiable or sensitive patient scans.**
