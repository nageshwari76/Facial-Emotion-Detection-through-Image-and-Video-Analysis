# 😊 EmoCare AI — Facial Emotion Detection Through Image & Video Analysis

An intelligent, web-based application that **detects human emotions in real time** from a **webcam** or an **uploaded/dropped image**, and then responds with **personalized motivational messages, emotional support, stress-relief suggestions and activity recommendations** — powered by your own **Gemini or ChatGPT API key** (or built-in fallback messages when no key is set).

- **Backend:** Django (Python)
- **Frontend:** HTML, CSS, JavaScript (fully responsive, light theme)
- **AI:** OpenCV (face detection) + pre-trained **Vision Transformer (ViT)** deep learning model (`best_fer2013_vit.h5`, **~70% accuracy** on the FER2013 test set) classifying **7 emotions**: Happy 😊, Sad 😢, Angry 😠, Fear 😨, Surprise 😲, Disgust 🤢, Neutral 😐
- **Dataset:** FER2013 (already present in the `dataset/` folder, `train/` + `test/` splits)
- **Training:** the admin panel can **retrain the MobileNetV2 model** from the notebook (`fer2013_mobilenetv2.ipynb`) or **evaluate any h5** — with matplotlib graphs (bar chart, confusion matrix, pie chart) and a classification report.

---

## ✨ Features

| Page | Description |
|---|---|
| 🏠 **Main Home Page** | Project theme, features, the 7 emotions and how-it-works overview |
| 👤 **User Registration** | New accounts start **inactive** until the admin approves them |
| 🔐 **User Login** | Login for approved users; pending accounts see a clear message |
| 🎥 **User Home** | **Live webcam detection** + **image upload** (file picker **or drag & drop**). The **captured image is shown with the prediction below it**; **every face** in the image gets its own labeled box. Confidence score, personalized AI message, detection history |
| 🛡️ **Admin Login** | Separate admin login (`admin` / `admin`) |
| 📊 **Admin Home** | Dashboard stats, **user management** (activate / deactivate / delete), **dataset view** (train & test counts per emotion), **API settings** |
| 🤖 **Train Model Tab** | Retrain the MobileNetV2 model per the notebook **or** evaluate any h5 — with live progress, matplotlib graphs and a classification report |
| 🔑 **Blank API Option** | Admin pastes their **own Gemini or ChatGPT API key** — the app then generates the messages; built-in random fallback messages are used when no key is set, the call fails, or the system is offline |

---

## 🧠 The Two Models

The project ships with **two pre-trained models**:

| Model | Role | Accuracy (FER2013 test) | Size | Speed |
|---|---|---|---|---|
| **`best_fer2013_vit.h5`** (Vision Transformer) | **Used by the app for detection** (`INFERENCE_MODEL_PATH`) | **~70%** | 328 MB | ~0.4 s/detection on CPU |
| `best_mobilenetv2_fer2013.h5` (MobileNetV2) | Retraining target (`MODEL_PATH`) — what **🚀 Run Model** writes to | varies (~32% after a short run) | 26 MB | fast |

Retraining **never overwrites** the ViT file — the admin "Run Model" always saves to `best_mobilenetv2_fer2013.h5`, and the app automatically falls back to it only if the ViT file is missing.

## 👁️ Face Detection Pipeline

### Invalid images → "No face image detected"

Every image is first checked by **YuNet**, OpenCV's CNN face detector
(`face_models/face_detection_yunet_2023mar.onnx`, built into `cv2.FaceDetectorYN`, no extra
packages). If it finds no face with confidence ≥ 0.7, no emotion is predicted: the result is
**"No face image detected"** (🚫), and it is saved in the history under that name. Uploading
a mobile phone, a plant, a chart, a hand, etc. therefore no longer returns a made-up emotion.

Measured here: 0 of 38 non-face images accepted (charts, hand photos, objects), 101 of 105
FER2013 faces accepted (the tiny 48×48 test images; normal photos and webcam captures were
all accepted). YuNet's face box is also used for the crop, which raised emotion accuracy on
that sample from 62% to 73%.

If the YuNet model file is missing, the app falls back to the Haar cascades below.

### Multiple faces in one image

When the camera capture or uploaded photo contains more than one person, **every face is
analyzed**, not just one:

1. All detected faces are cropped (with the 18% margin) and predicted by the model in a single batch.
2. Faces are numbered **left to right** (1, 2, 3 …).
3. The server returns a copy of the image with a **colored box and label on each face**,
   e.g. `1: Happy 90%`, `2: Sad 36%` (box colour depends on the emotion).
4. The result panel shows that labeled image, a **"N faces detected"** badge and a
   **per-face list** (Face 1, Face 2 …) whose numbers match the boxes on the image.
5. The **main face** — the largest one, usually the person closest to the camera — is
   highlighted in the list. Its emotion is the one shown in large text, used for the
   support message and saved in the detection history.

The `/analyze/` JSON response therefore includes, in addition to `emotion`, `confidence`
and `message` (main face):

| Field | Content |
|---|---|
| `faces` | `[{id, emotion, confidence}]` for every face, left to right |
| `main_face_id` | `id` of the largest face |
| `annotated_image` | `data:image/jpeg;base64,...` image with every face labeled |

The labeled image is also what gets **saved in the detection history**, and the
*Recent Detections* list shows it as a thumbnail (click to open it full size).

`/analyze/` only works for **logged-in users** and rejects uploads larger than **10 MB**.

### Haar cascade fallback

Detection quality was tuned for hard cases (small thumbnails, low light, slight tilts):

1. **Two Haar cascades** (`frontalface_default` + `frontalface_alt2`) tried together — they fail on different images
2. **Progressively relaxed parameters** (strict → loose)
3. **Three image variants** escalated automatically: plain → CLAHE (low-light fix) → global histogram equalization
4. **Multi-scale upscaling** (2×, 3×, 4×) for tiny images — Haar misses faces below ~24 px
5. **18% margin** added around the face box so the crop resembles FER2013 training data

Result on the FER2013 test set (48×48 images — the hardest case): **~94% face-detection rate**, up from ~70% with a single-cascade setup. Real webcam photos detect at nearly 100%.

## 🤖 Admin Panel — Train Model Tab

The admin panel has a **🤖 Train Model** tab with two buttons:

| Button | What it does | Speed |
|---|---|---|
| **⚡ Use Existing h5** | Loads the selected h5, runs one prediction pass over the test set (no training) and generates the graphs + report | Fast (seconds to a few minutes for the 328 MB ViT on CPU) |
| **🚀 Run Model** | Full retraining of the MobileNetV2 per `fer2013_mobilenetv2.ipynb` — two-stage transfer learning (feature extraction → fine-tuning) with balanced class weights. Saves to `best_mobilenetv2_fer2013.h5` | Slow on CPU (first run downloads ImageNet weights ~14 MB) |

Both run in a **background thread** with an animated progress bar; the admin page polls `/admin-training-status/` and reloads automatically when the job finishes.

### Graphs shown after a run (matplotlib / seaborn)

1. **📈 Training & Validation Accuracy/Loss curves** — *training runs only* (this data lives in training history, so it does not appear when using ⚡ Use Existing h5)
2. **🔥 Confusion matrix** heatmap on the test set
3. **📊 Class distribution bar chart** (Train vs Test)
4. **🥧 Class distribution pie chart**
5. **📋 Classification report** — precision / recall / F1 per emotion + test accuracy

The trained weights are saved to `best_mobilenetv2_fer2013.h5` (the same file the app loads) and the app automatically reloads the new model on the next detection.

---

## 🔑 How the API Messages Work (online / offline)

The app **never returns a blank message**. The logic in `emotion_app/utils/api_service.py`:

| Situation | Result |
|---|---|
| No API key set (or key removed) | Random built-in message for the detected emotion (works offline) |
| Online + valid key | Personalized message from Gemini / ChatGPT |
| Online + key, but the API fails | Falls back to a random built-in message |
| **Offline** (with or without key) | The API call times out fast (**8 s**) and falls back to a random built-in message |

---

## 🛠️ Tech Stack

- Python 3.10+ & Django 4.2+
- TensorFlow / Keras (ViT-Base FER model for detection, MobileNetV2 for retraining)
- OpenCV (`opencv-python`) — face detection & image processing
- NumPy, Pillow, Requests
- matplotlib, seaborn, scikit-learn — training graphs, confusion matrix, classification report
- HTML5 / CSS3 / Vanilla JavaScript (no build step needed)

---

## 📁 Project Structure

```
face_project/
├── manage.py
├── requirements.txt
├── README.md
├── db.sqlite3                      (created after migrate)
├── best_fer2013_vit.h5             (ViT model — USED by the app for detection, ~70% accuracy)
├── best_mobilenetv2_fer2013.h5     (MobileNetV2 — retraining target, written by "Run Model")
├── fer2013_mobilenetv2.ipynb       (training notebook the admin panel mirrors)
├── dataset.rar                     (archive copy of the FER2013 dataset)
├── dataset/                        (FER2013 dataset - provided)
│   ├── train/{angry,disgust,fear,happy,neutral,sad,surprise}
│   └── test/{angry,disgust,fear,happy,neutral,sad,surprise}
├── emotion_project/
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py / asgi.py
├── emotion_app/
│   ├── models.py / forms.py / views.py / urls.py / admin.py
│   ├── migrations/                 (incl. auto admin creation)
│   ├── templatetags/emotion_extras.py
│   ├── utils/
│   │   ├── emotion_model.py        (ViT inference for every face + face detection + labeled image)
│   │   ├── api_service.py          (Gemini / ChatGPT + fallback messages)
│   │   ├── train_model.py          (background training / evaluation pipeline)
│   │   └── vit_layers.py           (custom Keras layers required to load best_fer2013_vit.h5)
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/main.js, camera.js, admin_training.js
│   └── templates/emotion_app/*.html
├── media/
│   ├── (captured/uploaded images)
│   └── training/
│       ├── status.json             (live job status - polled by the admin panel)
│       ├── training_curves.png     (matplotlib graphs)
│       ├── confusion_matrix.png
│       ├── class_distribution.png
│       └── class_distribution_pie.png
```

---

## 🚀 Setup & Run (Windows)

> Bash/Git is **not** required — use the Anaconda Prompt, CMD, or any terminal.

### 1. Install dependencies

```bash
cd Desktop\face_project
pip install -r requirements.txt
```

> TensorFlow can take a while to install. If you already have a TensorFlow/Keras environment, you can skip the `tensorflow` line.

### 2. Create the database & admin user

```bash
python manage.py makemigrations emotion_app
python manage.py migrate
```

The migration automatically creates the admin account (**username:** `admin`, **password:** `admin`).

### 3. Start the server

```bash
python manage.py runserver
```

Open **http://127.0.0.1:8000/** in your browser.

### 4. (Optional) Plug in your own API key

1. Log in at **Admin Login** (`admin` / `admin`).
2. Go to the **🔑 API Settings** tab.
3. Choose **Google Gemini** or **OpenAI ChatGPT**, paste **your own API key** and enable it.
   - Gemini keys: https://aistudio.google.com/apikey
   - OpenAI keys: https://platform.openai.com/api-keys
4. Save. New detections will now return messages generated by *your* API (built-in fallbacks are used otherwise).

### 5. (Optional) Evaluate or retrain the model

1. Admin panel → **🤖 Train Model** tab.
2. Click **⚡ Use Existing h5** for a fast evaluation + graphs from the current model, or **🚀 Run Model** to retrain per the notebook.
3. Wait for the progress bar; the page reloads and shows the graphs + classification report.

---

## 👨‍💻 Usage Guide

1. **Register** a user account → it appears as **Pending** in the admin panel.
2. **Admin** → *Users* tab → click **Activate**.
3. The user can now **Login** → *User Dashboard*.
4. Click **Start Camera** and **Capture**, or **upload / drag & drop an image**.
5. The captured/uploaded image appears in the **Detection Result** panel with the prediction below it: emotion, confidence bar and a personalized message (from your API key, or the built-in fallback).
6. If there are **several people** in the picture, each face gets a numbered box with its emotion on the image, and a list below shows the prediction for every face (the largest face is the main result).

> 💡 The first detection after starting the server takes a few seconds — the 328 MB ViT model is loaded lazily once and cached.

---

## 🔑 Default Credentials

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin` |

> ⚠️ Change the admin password after first login! (`python manage.py changepassword admin`)

---

## 🧪 Manual Test Checklist

- [ ] Home page loads and is responsive (resize browser / open on phone)
- [ ] Register a user → admin panel shows it as **Pending**
- [ ] Login blocked with "not activated yet" message until admin activates
- [ ] Admin activate → user can log in
- [ ] Webcam detection returns an emotion + confidence + message
- [ ] The captured image is shown in the result panel with the prediction below it
- [ ] A photo with 2+ people shows a labeled box on every face, the "N faces detected" badge and the per-face list
- [ ] Image upload **and drag & drop** work the same way
- [ ] Detection history is saved under the user dashboard, with a thumbnail of the labeled image
- [ ] Admin can deactivate / delete users
- [ ] Dataset tab shows train/test counts for all 7 emotions
- [ ] API settings saved and messages switch to API-generated text
- [ ] **Train Model** tab: ⚡ Use Existing h5 evaluates the selected h5 and shows the bar chart, confusion matrix, pie chart + classification report
- [ ] **Train Model** tab: 🚀 Run Model retrains the MobileNetV2 (saves to `best_mobilenetv2_fer2013.h5`, never touches the ViT file)

---

## 🐛 Troubleshooting

- **"Model file not found"** → make sure `best_fer2013_vit.h5` (detection) and/or `best_mobilenetv2_fer2013.h5` (retraining) are in the project root. The app falls back to the MobileNet file if the ViT is missing.
- **Model fails to load (Keras / TensorFlow version error)** → the admin **Train Model** tab shows the exact error; the models here were saved with TensorFlow 2.21 / Keras 3.15. For the ViT, `vit_layers.py` must be importable (it is imported automatically before loading).
- **"Could not access the camera"** → allow camera permission in the browser, or use image upload.
- **First detection is slow (~5–10 s)** → normal, the 328 MB ViT model is loaded lazily once; later detections take under a second.
- **No face detected** → ensure the face is clearly visible and well lit. Detection now escalates through CLAHE / histogram equalization and 2×–4× upscaling automatically, but extremely small or heavily rotated faces can still be missed.
- **API message not appearing** → check the key is valid, the provider is selected, and *Enabled* is ticked (fallback messages are used otherwise).
- **Training graphs not updating** → wait for the run to finish; the page reloads automatically. If the server was restarted mid-run, the status is reset to *failed* — just click the button again.
- **Port already in use** → `python manage.py runserver 8001`

---

© EmoCare AI — Built with Django, OpenCV & Deep Learning.

---

## 📈 Model Benchmark (measured on this machine, FER2013 test set)

| Check | MobileNetV2 (`best_mobilenetv2_fer2013.h5`) | ViT (`best_fer2013_vit.h5`) |
|---|---|---|
| Full test accuracy (7,178 imgs, no face detection) | 31.6% | **70.6%** (700-img stratified sample) |
| Face detection rate (48×48 imgs) | — | **93.8%** (was 70% before tuning) |
| End-to-end (detect face + correct emotion) | — | **60.0%** on hardest images |

The MobileNetV2 checkpoint currently in the repo was overwritten by a short
retraining run — use the admin **🚀 Run Model** with plenty of epochs to produce
a fast fallback model, or just rely on the ViT for detection.

---

## 📝 Setup notes

- **Python:** 3.10 or newer (tested with Python 3.11).
- **Dependencies:** `pip install -r requirements.txt` (a virtual environment is recommended).
- **Paths:** every file path in the code is relative to the project folder, so the project runs from any location — just `cd` into this folder before running the commands above.
- **Time zone:** `TIME_ZONE = "Asia/Kolkata"` (IST) in `emotion_project/settings.py`, so all dates and times are shown in Indian Standard Time.


---

<!-- mobile-check 2026-10-02 -->
## 📱 Using it on a phone

All pages were checked at 360 px and 390 px wide (guest, user and admin views) on 2026-10-02: no sideways scrolling, menus usable, nothing cut off.

- Form fields use 16 px text on phones, so iPhones don't zoom in when a field is tapped.
- The stylesheet link has a `?v=20261002` suffix; change it after editing the CSS so phones don't keep a cached copy.
- **Webcam features need a secure page:** browsers only allow the camera on `https://` or `localhost`, so on `http://<PC-IP>:8000` a phone shows a 'camera not supported / page not secure' message — image/video upload still works. For a demo on Android Chrome, add the address under `chrome://flags` → *Insecure origins treated as secure*.

While `DEBUG` is on, any host name is accepted, so `python manage.py runserver 0.0.0.0:8000` and `http://<PC-IP>:8000` on a phone on the same Wi-Fi just works (find the IP with `ipconfig`).
