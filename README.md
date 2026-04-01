# COVID-19 & Pneumonia CT Scan Classification

CT scan image classification of COVID-19, Pneumonia, and Normal cases using **DenseNet121** feature extraction with SVM, Random Forest, and Logistic Regression classifiers.

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
Download from [Google Drive](https://drive.google.com/drive/folders/1ibOUHamULECObSAR1QDvZLw7C_PWZdg8?usp=sharing) (~3GB)

Place at: `/content/drive/MyDrive/DS1/Mendaly/`

## How to Run
1. Open notebook in Google Colab
2. Mount Google Drive
3. Download and place dataset at the path above
4. Run all cells

## Dependencies
```
tensorflow, opencv-python, scikit-learn, numpy, matplotlib, seaborn
```
