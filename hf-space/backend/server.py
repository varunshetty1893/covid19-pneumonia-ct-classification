"""Prediction API. Also serves ../frontend for local use.  Run from this folder:  python server.py"""
import os, json, cv2, joblib, numpy as np
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

here = os.path.dirname(os.path.abspath(__file__))
def first_dir(*c): return next((p for p in c if os.path.isdir(p)), None)
art = first_dir(os.path.join(here, "artifacts"), os.path.join(here, "..", "artifacts")) or os.path.join(here, "artifacts")
web = first_dir(os.path.join(here, "..", "frontend"))   # absent on a deployed API-only backend
LABELS = ["COVID-19", "Pneumonia", "Normal"]
NAMES = {"svm": "SVM", "rf": "Random Forest", "lr": "Logistic Regression"}
clfs = {k: joblib.load(os.path.join(art, k + ".joblib")) for k in NAMES if os.path.exists(os.path.join(art, k + ".joblib"))}
if not clfs: raise SystemExit("No trained models found in ./artifacts. Run: python train.py")
mp = os.path.join(art, "metrics.json"); metrics = json.load(open(mp)) if os.path.exists(mp) else {}
net = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
@app.after_request
def cors(r): r.headers["Access-Control-Allow-Origin"] = "*"; return r

@app.route("/")
def index():
    if web: return send_from_directory(web, "index.html")
    return jsonify(service="LungLens API", health="/health", predict="POST /predict (form field: image)")
@app.route("/<p>")
def static_files(p):
    if web and p in ("style.css", "app.js", "config.js", "index.html"): return send_from_directory(web, p)
    return "", 404
@app.get("/health")
def health(): return jsonify(ok=True, models=list(clfs))

@app.post("/predict")
def predict():
    f = request.files.get("image")
    if not f: return jsonify(error="No image received"), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: return jsonify(error="That file is not a readable image"), 400
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
    x = preprocess_input(np.expand_dims(img.astype("float32"), 0))
    feat = net.predict(x, verbose=0).reshape(1, -1)
    res = []
    for k, c in clfs.items():
        p = c.predict_proba(feat)[0]; i = int(np.argmax(p))
        res.append(dict(model=NAMES[k], label=LABELS[i], confidence=float(p[i]),
                        probs={LABELS[j]: float(p[j]) for j in range(3)}, test_accuracy=metrics.get(k)))
    return jsonify(results=res)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5173)); print(f"Open http://localhost:{port}")
    app.run(host="0.0.0.0", port=port)
