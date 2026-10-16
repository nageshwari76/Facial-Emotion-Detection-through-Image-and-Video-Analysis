"""Facial emotion recognition service.

Loads the high-accuracy ViT model (``best_fer2013_vit.h5``) lazily for
detection, finds every face with YuNet (Haar cascades as a fallback), predicts
the emotion of each face and returns a copy of the image with every face
labeled. The ViT model reaches ~70% on the FER2013 test set (vs ~32% for the
smaller MobileNetV2 checkpoint), so it is the one used by the app.

Preprocessing mirrors the model's training setup: RGB images are resized to
the model input size and scaled with ``mobilenet_v2.preprocess_input``
(values in ``[-1, 1]``) — the ViT uses the same [-1, 1] scaling, not a plain
``/255`` rescale.
"""
import logging
import os
import threading

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

# Order of the 7 FER2013 emotion classes used by the model.
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Result shown (and stored in history) when the image contains no human face,
# e.g. a mobile phone, a plant or a chart.
NO_FACE_EMOTION = 'no_face'
NO_FACE_LABEL = 'No face image detected'

# YuNet (OpenCV's CNN face detector) validates that an image really contains a
# face. The old Haar cascades - especially with the relaxed retry settings
# below - "found" faces in almost any picture (28 of 29 test photos of charts,
# hands and objects), so objects were given an emotion. YuNet kept every real
# face in testing while rejecting those objects.
FACE_SCORE_THRESHOLD = 0.7

_model = None
_model_lock = threading.Lock()
_face_cascades = None
_yunet = None
_yunet_checked = False


def get_model():
    """Load (once) and return the Keras model used for detection.

    Uses ``INFERENCE_MODEL_PATH`` (the high-accuracy ViT model) when present,
    falling back to ``MODEL_PATH`` (the MobileNetV2 checkpoint) otherwise.
    """
    global _model
    if _model is not None:
        return _model
    # The dev server is multi-threaded: without the lock, two simultaneous
    # first requests would each load the 328 MB model.
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            raise ImportError(
                'TensorFlow is not installed. Run: pip install tensorflow'
            )
        # Register custom ViT layers (ClsToken, AddPosEmbed, ...) so the h5
        # file with custom objects can load.
        try:
            from . import vit_layers  # noqa: F401
        except ImportError:
            pass
        # Prefer the ViT inference model; fall back to the MobileNet file.
        inference_path = getattr(settings, 'INFERENCE_MODEL_PATH', None)
        if inference_path is not None and inference_path.exists():
            path = str(inference_path)
        else:
            path = str(settings.MODEL_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Model file not found at {path}. '
                'Make sure best_fer2013_vit.h5 (or best_mobilenetv2_fer2013.h5) is in the project folder.'
            )
        try:
            _model = load_model(path, compile=False)
        except Exception as exc:  # e.g. Keras version mismatches
            raise RuntimeError(
                f'Could not load the model file ({exc}). '
                'Try retraining the model from the admin panel.'
            )
    return _model


def reset_model_cache():
    """Drop the cached model so the next detection reloads it from disk.

    Called automatically after the admin retrains the model from the panel.
    """
    global _model
    _model = None


def get_face_cascades():
    """Load (once) the frontal-face Haar cascades used for detection.

    Two cascade variants are loaded: ``frontalface_default`` and
    ``frontalface_alt2``. They fail on different images, so trying both
    substantially raises the detection rate. Returns a non-empty list.
    """
    global _face_cascades
    if _face_cascades is None:
        cascades = []
        for name in ('haarcascade_frontalface_default.xml',
                     'haarcascade_frontalface_alt2.xml'):
            cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))
            if not cascade.empty():
                cascades.append(cascade)
        if not cascades:  # should not happen — cv2 ships both files
            cascades.append(cv2.CascadeClassifier(
                os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')))
        _face_cascades = cascades
    return _face_cascades


def _model_input_size(model):
    """Return (height, width, channels) of the model's first input.

    Works for both Sequential and Functional Keras models, where
    ``model.layers[0].input_shape`` can be a tuple or a list of TensorShapes.
    """
    # Most reliable: the actual input tensor of the model.
    try:
        shape = tuple(model.inputs[0].shape)  # (None, height, width, channels)
        return int(shape[1]), int(shape[2]), int(shape[3])
    except Exception:
        pass
    # Fallback for older saved models.
    raw = model.layers[0].input_shape
    if isinstance(raw, (list, tuple)) and raw and not isinstance(raw[0], int):
        raw = raw[0]
    return int(raw[1]), int(raw[2]), int(raw[3])


def preprocess_face(face_bgr):
    """Resize + normalize a face crop exactly like the notebook's generator.

    ``fer2013_mobilenetv2.ipynb`` feeds RGB images through
    ``tensorflow.keras.applications.mobilenet_v2.preprocess_input``, which
    rescales ``[0, 255]`` to ``[-1, 1]``. We reproduce that here. A 1-channel
    model (plain FER2013 CNN) falls back to grayscale ``/255`` scaling.
    """
    model = get_model()
    target_h, target_w, channels = _model_input_size(model)

    if channels == 1:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (target_w, target_h))
        arr = resized.reshape(target_h, target_w, 1)
        arr = arr.astype('float32') / 255.0
    else:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_w, target_h))
        # Match the notebook: mobilenet_v2.preprocess_input ([-1, 1] range).
        try:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            arr = preprocess_input(resized.astype('float32'))
        except Exception:
            arr = resized.astype('float32') / 255.0

    return np.expand_dims(arr, axis=0)


_DETECT_PARAMS = [
    # Progressively relaxed Haar parameters — strict first, loose last.
    {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (48, 48)},
    {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (40, 40)},
    {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (32, 32)},
    {'scaleFactor': 1.03, 'minNeighbors': 2, 'minSize': (24, 24)},
]


def _run_cascades(gray):
    """Try every cascade × parameter combination on one grayscale image.

    Returns a list of (x, y, w, h) rectangles (may be empty).
    """
    for cascade in get_face_cascades():
        for params in _DETECT_PARAMS:
            faces = cascade.detectMultiScale(gray, **params)
            if len(faces) > 0:
                return faces
    return []


def _clahe(gray):
    """Contrast-limited adaptive histogram equalization (helps low light)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _detect_faces(gray):
    """Detect faces on a grayscale image, escalating through three image
    variants: plain → CLAHE → global histogram equalization.

    Returns a list of (x, y, w, h) rectangles in ``gray`` coordinates.
    """
    faces = _run_cascades(gray)
    if len(faces) > 0:
        return faces
    # Low-contrast / uneven lighting: normalize locally, then globally.
    faces = _run_cascades(_clahe(gray))
    if len(faces) > 0:
        return faces
    return _run_cascades(cv2.equalizeHist(gray))


def _detect_faces_multiscale(gray):
    """Detect faces, retrying on upscaled copies for small images.

    Haar cascades miss faces below roughly 24 px, so tiny images (e.g. the
    48×48 FER2013 pictures or thumbnails) are upscaled 2×, 3×, 4× until a
    face is found. Returned rectangles are mapped back to original coords.
    """
    faces = _detect_faces(gray)
    if len(faces) > 0:
        return faces

    h, w = gray.shape
    for scale in (2, 3, 4):
        if max(h, w) * scale > 4000:  # cap memory use
            break
        upscaled = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        found = _detect_faces(upscaled)
        if len(found) > 0:
            faces = [
                (int(x / scale), int(y / scale), int(fw / scale), int(fh / scale))
                for (x, y, fw, fh) in found
            ]
            break
    return faces


def _yunet_path():
    default = os.path.join(str(settings.BASE_DIR), 'face_models', 'face_detection_yunet_2023mar.onnx')
    return str(getattr(settings, 'FACE_DETECTOR_MODEL_PATH', default))


def get_face_validator():
    """Load (once) the YuNet face detector, or return None if the model file
    is missing (the app then falls back to the Haar cascades)."""
    global _yunet, _yunet_checked
    if not _yunet_checked:
        _yunet_checked = True
        path = _yunet_path()
        if os.path.exists(path) and hasattr(cv2, 'FaceDetectorYN'):
            try:
                _yunet = cv2.FaceDetectorYN.create(path, '', (320, 320), FACE_SCORE_THRESHOLD)
            except Exception:
                logger.exception('Could not load the YuNet face detector')
                _yunet = None
        else:
            logger.warning('YuNet face model not found at %s - using Haar cascades only', path)
    return _yunet


def _detect_faces_yunet(img, detector):
    """Return [(x, y, w, h, score)] for faces YuNet is confident about.

    Small images (e.g. 48x48 FER2013 crops) are upscaled and very large ones
    downscaled so faces fall in the range the detector handles well; boxes are
    mapped back to original image coordinates.
    """
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) < 320:
        scale = 320.0 / max(h, w)
    elif max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
    work = img
    if scale != 1.0:
        work = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    detector.setInputSize((work.shape[1], work.shape[0]))
    _, found = detector.detect(work)
    faces = []
    if found is not None:
        for row in found:
            score = float(row[14])
            if score < FACE_SCORE_THRESHOLD:
                continue
            x, y, fw, fh = (float(v) / scale for v in row[:4])
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w, int(x + fw)), min(h, int(y + fh))
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                faces.append((x0, y0, x1 - x0, y1 - y0, score))
    return faces


def no_face_result():
    return {'emotion': None, 'confidence': 0.0, 'no_face': True, 'error': NO_FACE_LABEL}


def _expand_face_box(x, y, w, h, img_w, img_h, margin=0.18):
    """Grow a face box by ``margin`` on every side (clamped to the image).

    FER2013 training crops include hairline/chin context, while Haar boxes
    are tight on the face — the padding makes real-world crops resemble the
    training data, which measurably improves predictions.
    """
    dx = int(w * margin)
    dy = int(h * margin)
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(img_w, x + w + dx)
    y1 = min(img_h, y + h + dy)
    return x0, y0, x1 - x0, y1 - y0


# BGR box colours per emotion for the annotated image.
EMOTION_COLORS = {
    'angry': (40, 40, 220), 'disgust': (40, 140, 40), 'fear': (160, 60, 160),
    'happy': (0, 190, 255), 'neutral': (160, 160, 160), 'sad': (200, 110, 30),
    'surprise': (180, 180, 0),
}


def _annotate_faces(img, faces):
    """Draw a numbered box + "N: Emotion XX%" label on every face.

    Returns the annotated image as JPEG bytes.
    """
    out = img.copy()
    img_h, img_w = out.shape[:2]
    thickness = max(2, int(round(max(img_h, img_w) / 400)))
    font_scale = max(0.5, max(img_h, img_w) / 1000)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for face in faces:
        x, y, w, h = face['box']
        color = EMOTION_COLORS.get(face['emotion'], (0, 200, 0))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        label = f"{face['id']}: {face['emotion'].capitalize()} {face['confidence'] * 100:.0f}%"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        # Label above the box, or inside its top edge when there is no room.
        top = y - th - baseline - 6
        if top < 0:
            top = y
        cv2.rectangle(out, (x, top), (x + tw + 8, top + th + baseline + 6), color, -1)
        b, g, r = color
        text_color = (0, 0, 0) if 0.299 * r + 0.587 * g + 0.114 * b > 150 else (255, 255, 255)
        cv2.putText(out, label, (x + 4, top + th + 3), font, font_scale,
                    text_color, thickness, cv2.LINE_AA)
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else None


def analyze_face_image(img_bytes):
    """Detect every face in the image and predict each one's emotion.

    Returns a dict with keys:
      emotion, confidence - the largest (main) face, kept for the history record
      faces               - [{id, box, emotion, confidence}] ordered left to right
      annotated_image     - JPEG bytes with a labeled box drawn on every face
      error
    """
    try:
        model = get_model()
    except Exception as exc:
        logger.exception('Model could not be loaded')
        return {'emotion': None, 'confidence': 0.0, 'error': str(exc)}

    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        return {'emotion': None, 'confidence': 0.0, 'error': 'Could not read the image. Try another image.'}

    validator = get_face_validator()
    if validator is not None:
        # CNN face check: objects, plants, charts, hands ... -> "No face image detected".
        found = _detect_faces_yunet(img, validator)
        if not found:
            return no_face_result()
        faces = [f[:4] for f in found]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _detect_faces_multiscale(gray)
        if len(faces) == 0:
            return no_face_result()

    # Number faces left to right so labels on the image match the result list.
    boxes = sorted((tuple(int(v) for v in f) for f in faces), key=lambda b: b[0])

    try:
        # Each face is padded to match FER2013-style crops; all are predicted in one batch.
        crops = []
        for (x, y, w, h) in boxes:
            ex, ey, ew, eh = _expand_face_box(x, y, w, h, img.shape[1], img.shape[0])
            crops.append(preprocess_face(img[ey:ey + eh, ex:ex + ew]))
        all_preds = model.predict(np.concatenate(crops, axis=0), verbose=0)

        results = []
        for i, (box, preds) in enumerate(zip(boxes, all_preds), start=1):
            idx = int(np.argmax(preds))
            if int(preds.shape[0]) == len(EMOTIONS):
                emotion = EMOTIONS[idx]
            else:
                emotion = f'class_{idx}'  # fallback for unexpected output sizes
            results.append({'id': i, 'box': list(box), 'emotion': emotion,
                            'confidence': float(preds[idx])})

        main = max(results, key=lambda r: r['box'][2] * r['box'][3])
        return {
            'emotion': main['emotion'],
            'confidence': main['confidence'],
            'main_face_id': main['id'],
            'faces': results,
            'annotated_image': _annotate_faces(img, results),
            'error': None,
        }
    except Exception as exc:  # pragma: no cover - model/tensor errors surface to UI
        logger.exception('Emotion prediction failed')
        return {'emotion': None, 'confidence': 0.0, 'error': f'Prediction failed: {exc}'}
