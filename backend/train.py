"""Trains the 3 classifiers from your notebook and saves them for the website.
Run from this folder:   python train.py
Looks for the dataset in ./Mendaly (folders COVID2_CT, pneumonia_CT, Normal_CT).
Optional:               python train.py "D:/some/other/Mendaly"
"""
import os, sys, json, cv2, joblib, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

here = os.path.dirname(os.path.abspath(__file__))
default = next((p for p in (os.path.join(here, "Mendaly"), os.path.join(here, "..", "Mendaly")) if os.path.isdir(p)), os.path.join(here, "Mendaly"))
base = sys.argv[1] if len(sys.argv) > 1 else default
classes = {"COVID2_CT": 0, "pneumonia_CT": 1, "Normal_CT": 2}

# Handle a nested copy such as Mendaly/Mendaly/COVID2_CT
if not os.path.isdir(os.path.join(base, "COVID2_CT")) and os.path.isdir(os.path.join(base, "Mendaly")):
    base = os.path.join(base, "Mendaly")
missing = [c for c in classes if not os.path.isdir(os.path.join(base, c))]
if missing:
    sys.exit(f"Dataset not found in: {base}\nMissing folders: {missing}\nExpected COVID2_CT, pneumonia_CT, Normal_CT inside it.")

art = os.path.join(here, "artifacts"); os.makedirs(art, exist_ok=True)
cache = os.path.join(art, "features.npz")

if os.path.exists(cache):
    print("Using cached features (delete artifacts/features.npz to re-extract)")
    d = np.load(cache); X, y = d["X"], d["y"]
else:
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.applications.densenet import preprocess_input
    paths, labels = [], []
    for name, label in classes.items():
        for f in sorted(os.listdir(os.path.join(base, name))):
            paths.append(os.path.join(base, name, f)); labels.append(label)
    print(f"Found {len(paths)} files. Extracting DenseNet121 features (about 10-20 min on CPU)...")
    net = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    feats, ys, skipped = [], [], 0
    for s in range(0, len(paths), 32):
        batch, yb = [], []
        for i in range(s, min(s + 32, len(paths))):
            img = cv2.imread(paths[i])
            if img is None: skipped += 1; continue
            batch.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))); yb.append(labels[i])
        if batch:
            out = net.predict(preprocess_input(np.array(batch, dtype="float32")), verbose=0)
            feats.append(out.reshape(len(batch), -1).astype("float32")); ys += yb
        print(f"  {min(s + 32, len(paths))}/{len(paths)} images", end="\r")
    X, y = np.vstack(feats), np.array(ys)
    np.savez(cache, X=X, y=y)
    print(f"\nUsed {len(y)} images, skipped {skipped} unreadable.")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clfs = {"svm": SVC(kernel="linear", probability=True),
        "rf": RandomForestClassifier(n_estimators=100),
        "lr": LogisticRegression(max_iter=1000)}
metrics = {}
for k, c in clfs.items():
    print(f"Training {k}...")
    c.fit(Xtr, ytr)
    metrics[k] = float(accuracy_score(yte, c.predict(Xte)))
    joblib.dump(c, os.path.join(art, k + ".joblib"))
    print(f"  {k} test accuracy: {metrics[k]:.4f}")
json.dump(metrics, open(os.path.join(art, "metrics.json"), "w"))
print("\nDone. Models saved in ./artifacts. Now run:  python server.py")
