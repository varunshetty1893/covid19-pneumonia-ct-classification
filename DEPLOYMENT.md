# Deploy LungLens: Hugging Face Gradio Space + Vercel

This package has a static Vercel frontend and a Gradio API hosted in a Hugging Face Space. Deploy the Space first, then configure the Vercel frontend with the Space ID.

## Before deployment: prepare trained model files

The supplied source zip does not contain the trained models or the approximately 3 GB dataset. Train the models locally or provide previously trained files. The API requires:

```text
backend/artifacts/svm.joblib
backend/artifacts/rf.joblib
backend/artifacts/lr.joblib
backend/artifacts/metrics.json
```

The training instructions and dataset link are in `README.md`. Do not upload the dataset to Hugging Face or GitHub.

## 1. Create a Gradio Space

1. On Hugging Face, choose **New Space**.
2. Select **Gradio** as the SDK, then choose **ZeroGPU (Free)** hardware, as shown in the setup screen.
3. Set the Space to **Public** so the Vercel browser client can call it without embedding a secret token.
4. If Hugging Face does not allow you to create a ZeroGPU Space, its current requirements for free personal accounts include a verified email, an account in good standing, and an account at least 30 days old. See Hugging Face's [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu).

**Important performance note:** the existing model uses `tensorflow-cpu` and Keras DenseNet121. This Gradio Space keeps the original trained model pipeline intact, but predictions run on CPU; ZeroGPU's dynamic accelerator is intended for PyTorch-based functions and is not used by this app. The free hardware can also have availability, queue, and sleep delays.

## 2. Upload the Space app and model files

Run these commands from the extracted project root in a terminal. Replace `<HF-USERNAME>` and `<SPACE-NAME>` with your Hugging Face account and Space name. Git, Git LFS, and Python must be installed.

Authenticate locally; do not share an access token in chat:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
git lfs install
```

Clone the new Space, copy the Gradio app bundle, add trained artifacts, and push:

```bash
git clone https://huggingface.co/spaces/<HF-USERNAME>/<SPACE-NAME> hf-space-deploy
cp -a hf-space/. hf-space-deploy/
mkdir -p hf-space-deploy/artifacts
cp backend/artifacts/svm.joblib backend/artifacts/rf.joblib backend/artifacts/lr.joblib backend/artifacts/metrics.json hf-space-deploy/artifacts/
cd hf-space-deploy
git lfs track "artifacts/*.joblib"
git add .
git commit -m "Deploy LungLens Gradio app"
git push
```

The Space bundle supplies `app.py`, Gradio metadata, requirements, and Git LFS tracking. On Windows, run the copy commands in Git Bash, or copy the contents of `hf-space/` into the cloned Space folder and copy the four artifact files into its `artifacts/` folder manually.

Wait for the Space build to finish. Open its Hugging Face page and confirm the Gradio app appears. The API endpoint is named `/predict` and accepts one image file.

## 3. Configure the Vercel frontend

Edit `frontend/config.js` in the project and set `window.CT_SPACE` to your Space ID (`username/space-name`):

```js
window.CT_API = "";
window.CT_SPACE = "<HF-USERNAME>/<SPACE-NAME>";
```

Leave `CT_API` empty for the Gradio deployment. The frontend uses the official Gradio JavaScript client, loaded from jsDelivr, to upload the scan and call `/predict`.

## 4. Deploy the frontend to Vercel

1. Push the project to GitHub. Do not include the dataset or trained `.joblib` files in the GitHub repository; the model artifacts are pushed to the Space separately.
2. In Vercel, import the GitHub repository.
3. Set **Root Directory** to `frontend`.
4. Choose **Other** as the Framework Preset. Leave the Build Command blank; use `.` as the Output Directory if Vercel asks for one.
5. Deploy. Pushes to the connected GitHub branch will redeploy the frontend.

## 5. Verify

Open the Vercel deployment, go to **Classify**, and confirm the status changes to **Classifier online**. Upload a JPG or PNG and run a classification. If the status stays offline, check the Space is public and running, then verify `CT_SPACE` exactly matches its `username/space-name`.

## Optional: UptimeRobot

For an uptime check, monitor the public Space's root URL (the Space page), not `/health`; this Gradio app does not expose the old Flask health route. Periodic checks may reduce idle time, but they do not guarantee always-on availability.

## Privacy and medical-use warning

The Space is public and has no authentication. Anyone can call its image-classification API. Do not send identifiable or sensitive patient scans. This project is for research and education only; it is not a medical diagnostic tool.
