"""Gradio wrapper for the existing LungLens TensorFlow/scikit-learn classifier."""
import json
from pathlib import Path

import cv2
import gradio as gr
import joblib
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
LABELS = ["COVID-19", "Pneumonia", "Normal"]
NAMES = {"svm": "SVM", "rf": "Random Forest", "lr": "Logistic Regression"}

missing = [
    f"{key}.joblib"
    for key in NAMES
    if not (ARTIFACTS / f"{key}.joblib").is_file()
]
if missing:
    raise RuntimeError(
        "Missing trained model files in artifacts/: "
        + ", ".join(missing)
        + ". Train the models with backend/train.py and upload the artifacts."
    )

classifiers = {
    key: joblib.load(ARTIFACTS / f"{key}.joblib")
    for key in NAMES
}
metrics_path = ARTIFACTS / "metrics.json"
metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}

# Same pretrained feature extractor and preprocessing as the original Flask API.
feature_net = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)


def predict(image_path):
    if not image_path:
        return {"error": "No image was received."}

    image = cv2.imread(str(image_path))
    if image is None:
        return {"error": "That file is not a readable image."}

    image = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        (224, 224),
    )
    tensor = preprocess_input(np.expand_dims(image.astype("float32"), axis=0))
    features = feature_net.predict(tensor, verbose=0).reshape(1, -1)

    results = []
    for key, classifier in classifiers.items():
        probabilities = classifier.predict_proba(features)[0]
        winner = int(np.argmax(probabilities))
        results.append(
            {
                "model": NAMES[key],
                "label": LABELS[winner],
                "confidence": float(probabilities[winner]),
                "probs": {
                    LABELS[index]: float(probabilities[index])
                    for index in range(len(LABELS))
                },
                "test_accuracy": metrics.get(key),
            }
        )

    return {"results": results}


with gr.Blocks(title="LungLens CT Classifier") as demo:
    gr.Markdown(
        "## LungLens CT Classifier\n"
        "Research and educational use only — this is not a medical diagnostic tool."
    )
    image = gr.Image(type="filepath", label="Upload a CT image")
    run = gr.Button("Classify")
    output = gr.JSON(label="Classifier results")
    run.click(fn=predict, inputs=image, outputs=output, api_name="predict")


if __name__ == "__main__":
    demo.launch()
