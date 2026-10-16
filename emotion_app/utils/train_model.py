"""Background model training pipeline.

Port of ``fer2013_mobilenetv2.ipynb`` so the admin can retrain the emotion
model straight from the admin panel:

- Two-stage MobileNetV2 transfer learning on the local ``dataset/`` folder.
- Balanced class weights (handles the small ``disgust`` class).
- Stage 1 feature extraction -> Stage 2 top-layer fine-tuning.
- Best weights saved to ``best_mobilenetv2_fer2013.h5`` (same file the app loads).
- Matplotlib graphs regenerated after each run (training curves, confusion
  matrix, class distribution) plus the test-set classification report.

Training runs in a background thread; progress and results are written to
``media/training/status.json`` so the admin panel can poll and display them
without blocking the request.
"""
import json
import threading
from datetime import datetime
from pathlib import Path

from django.conf import settings

# Reference to the currently running training thread (if any).
_training_thread = None

# Tunable hyper-parameters — optimised for CPU speed on Windows.
IMG_SIZE = (128, 128)       # 3x fewer pixels than 224 — much faster, still good
BATCH_SIZE = 32


def scan_model_files() -> list:
    """Scan the project root for .h5 model files and return a list of
    ``{'name': filename, 'path': full_path, 'size_mb': size}`` dicts,
    sorted by modification time (newest first).
    """
    import os
    base = Path(settings.BASE_DIR)
    files = []
    for f in base.glob('*.h5'):
        if f.is_file():
            size_mb = round(f.stat().st_size / (1024 * 1024), 1)
            files.append({
                'name': f.name,
                'path': str(f),
                'size_mb': size_mb,
                'mtime': f.stat().st_mtime,
            })
    files.sort(key=lambda x: x['mtime'], reverse=True)
    return files

PLOT_FILES = [
    'training/training_curves.png',
    'training/confusion_matrix.png',
    'training/class_distribution.png',
    'training/class_distribution_pie.png',
]


# ---------------------------------------------------------------------------
# Status persistence (media/training/status.json)
# ---------------------------------------------------------------------------

def get_status_dir() -> Path:
    return Path(settings.MEDIA_ROOT) / 'training'


def get_status_file() -> Path:
    return get_status_dir() / 'status.json'


def read_status() -> dict:
    """Return the current training status dict (never raises).

    If the status file says ``running`` but the worker thread is no longer
    alive (e.g. the server was restarted mid-run), the status is demoted to
    ``failed`` so the admin can start a fresh run.
    """
    try:
        data = json.loads(get_status_file().read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('bad status shape')
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        data = {}

    if data.get('status') == 'running' and (_training_thread is None or not _training_thread.is_alive()):
        data['status'] = 'failed'
        data['message'] = 'The job was interrupted (server restarted?). Click the button to run it again.'
        _write_status(data)
    return data


def _write_status(data: dict) -> None:
    get_status_dir().mkdir(parents=True, exist_ok=True)
    get_status_file().write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def update_status(**fields) -> None:
    """Merge ``fields`` into the status file (safe to call from the worker)."""
    data = read_status()
    data.update(fields)
    _write_status(data)


def start_training(epochs_stage1: int = 10, epochs_stage2: int = 3, model_file: str = ''):
    """Kick off training in a background thread.

    Returns ``(ok, message)`` so the view can flash a message.
    """
    if read_status().get('status') == 'running':
        return False, 'A job is already running. Please wait for it to finish.'
    return _start_job(_run_training, 'Model training started in the background. The page will update automatically.',
                      epochs_stage1, epochs_stage2, model_file)


def start_evaluation(model_file: str = ''):
    """Evaluate the existing ``best_mobilenetv2_fer2013.h5`` on the test set
    and regenerate the graphs — no training, much faster.

    Returns ``(ok, message)`` so the view can flash a message.
    """
    if read_status().get('status') == 'running':
        return False, 'A job is already running. Please wait for it to finish.'
    return _start_job(_run_evaluation, 'Evaluation started. Graphs will update automatically when it finishes.',
                      model_file)


def _start_job(target, ok_message, *args) -> tuple:
    """Shared runner for background jobs (training or evaluation)."""
    global _training_thread

    # Start the thread first (after .start() the thread is alive, so a
    # concurrent read_status() won't mistake it for a stale run), then write
    # the running state so the admin page reloading right after this request
    # sees the job as running. We write directly (not via update_status) to
    # avoid the demotion check in read_status.
    _training_thread = threading.Thread(
        target=target,
        args=args,
        daemon=True,
        name='emotion-model-job',
    )
    _training_thread.start()
    _write_status({
        'status': 'running',
        'message': 'Queued — starting...',
        'started_at': datetime.now().isoformat(timespec='seconds'),
        'finished_at': None,
        'plots': [],
        'report': '',
        'metrics': {},
        'progress': 1,
    })
    return True, ok_message


# ---------------------------------------------------------------------------
# Training pipeline (mirrors fer2013_mobilenetv2.ipynb)
# ---------------------------------------------------------------------------

def _run_training(epochs_stage1: int, epochs_stage2: int) -> None:
    started = datetime.now().isoformat(timespec='seconds')
    update_status(
        status='running',
        message='Initializing training...',
        started_at=started,
        finished_at=None,
        plots=[],
        report='',
        metrics={},
        progress=5,
    )
    try:
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.utils.class_weight import compute_class_weight
        from tensorflow.keras import layers, models, optimizers, callbacks
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
        except ImportError:  # newer TF/Keras moved the legacy preprocessing module
            from keras.preprocessing.image import ImageDataGenerator

        # Speed optimisations for CPU training.
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except Exception:
            pass  # not all CPUs support float16; safe to skip

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        dataset_dir = Path(settings.BASE_DIR) / 'dataset'
        train_dir = dataset_dir / 'train'
        test_dir = dataset_dir / 'test'
        for d in (train_dir, test_dir):
            if not d.is_dir():
                raise FileNotFoundError(f'Dataset folder not found: {d}')

        classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        if not classes:
            raise FileNotFoundError('No emotion class folders found in dataset/train.')
        update_status(status='running', message=f'Found {len(classes)} emotion classes.', progress=10)

        # ---- Class distribution + balanced class weights (notebook Step 3) ----
        train_counts = {}
        y_labels = []
        for idx, cls in enumerate(classes):
            n = len([f for f in (train_dir / cls).iterdir() if f.is_file()])
            train_counts[cls] = n
            y_labels.extend([idx] * n)
        y_labels = np.asarray(y_labels)
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_labels), y=y_labels)
        class_weight = dict(enumerate(weights))
        update_status(status='running', message='Class weights computed. Building data generators...', progress=20)

        # ---- Data generators (notebook Step 4) ----
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest',
        )
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_generator = train_datagen.flow_from_directory(
            str(train_dir), target_size=IMG_SIZE, color_mode='rgb',
            batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True,
        )
        test_generator = test_datagen.flow_from_directory(
            str(test_dir), target_size=IMG_SIZE, color_mode='rgb',
            batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False,
        )

        # ---- Model architecture (notebook Step 5) ----
        update_status(status='running', message='Building MobileNetV2 model (ImageNet weights)...', progress=30)
        base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        base_model.trainable = False

        inputs = layers.Input(shape=(128, 128, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(len(classes), activation='softmax')(x)
        model = models.Model(inputs, outputs, name='FER2013_MobileNetV2')

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')],
        )

        model_path = str(settings.MODEL_PATH)

        # Small callback that reports per-epoch progress to the status file.
        class _StatusCallback(tf.keras.callbacks.Callback):
            def __init__(self, stage, total):
                super().__init__()
                self.stage = stage
                self.total = total

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                logs = {k: round(float(v), 4) for k, v in logs.items() if isinstance(v, (int, float))}
                if self.stage == 1:
                    progress = 35 + int((epoch + 1) / max(self.total, 1) * 30)
                else:
                    progress = 70 + int((epoch + 1) / max(self.total, 1) * 15)
                update_status(
                    status='running',
                    message=(
                        f'Stage {self.stage}: epoch {epoch + 1}/{self.total} — '
                        f'acc {logs.get("accuracy", 0):.4f} | val_acc {logs.get("val_accuracy", 0):.4f}'
                    ),
                    stage=self.stage,
                    epoch=epoch + 1,
                    total_epochs=self.total,
                    last_logs=logs,
                    progress=min(progress, 84),
                )

        # ---- Stage 1: feature extraction (notebook Step 6) ----
        update_status(status='running', message=f'Stage 1: feature extraction ({epochs_stage1} epochs)...', progress=35)
        callbacks_stage1 = [
            callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0),
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
            _StatusCallback(stage=1, total=epochs_stage1),
        ]
        history_stage1 = model.fit(
            train_generator, epochs=epochs_stage1, validation_data=test_generator,
            class_weight=class_weight, callbacks=callbacks_stage1, verbose=0,
        )

        # ---- Stage 2: fine-tune top base layers (notebook Step 7) ----
        if epochs_stage2 > 0:
            update_status(status='running', message=f'Stage 2: fine-tuning top layers ({epochs_stage2} epochs)...', progress=70)
            base_model.trainable = True
            fine_tune_at = len(base_model.layers) - 40
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

            model.compile(
                optimizer=optimizers.Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy',
                         tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall')],
            )
            callbacks_stage2 = [
                callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=0),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0),
                callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=0),
                _StatusCallback(stage=2, total=epochs_stage2),
            ]
            history_stage2 = model.fit(
                train_generator, epochs=epochs_stage2, validation_data=test_generator,
                class_weight=class_weight, callbacks=callbacks_stage2, verbose=0,
            )
        else:
            history_stage2 = None
        model.save(model_path)

        # ---- Evaluation (notebook Step 8) ----
        update_status(status='running', message='Evaluating on the test set...', progress=85)
        test_generator.reset()
        predictions = model.predict(test_generator, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        report_text = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        test_accuracy = float(np.mean(y_pred == y_true))

        # ---- Matplotlib graphs (notebook Steps 8 & 9) ----
        out_dir = get_status_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        _plot_training_curves(history_stage1, history_stage2, out_dir / 'training_curves.png', plt)
        _plot_confusion_matrix(cm, classes, out_dir / 'confusion_matrix.png', plt, sns)
        _plot_class_distribution(train_counts, classes, test_dir, out_dir / 'class_distribution.png', plt, np)
        _plot_class_distribution_pie(train_counts, classes, out_dir / 'class_distribution_pie.png', plt)

        total_epochs = len(history_stage1.history['accuracy'])
        stage2_epochs = len(history_stage2.history['accuracy']) if history_stage2 else 0
        total_epochs += stage2_epochs
        update_status(
            status='completed',
            message=f'Training completed. Test accuracy: {test_accuracy:.4f} ({total_epochs} epochs total).',
            finished_at=datetime.now().isoformat(timespec='seconds'),
            progress=100,
            plots=PLOT_FILES,
            report=report_text,
            metrics={
                'test_accuracy': round(test_accuracy, 4),
                'stage1_epochs': len(history_stage1.history['accuracy']),
                'stage2_epochs': stage2_epochs,
            },
        )

        # Force the app to reload the freshly trained weights on next detection.
        from .emotion_model import reset_model_cache
        reset_model_cache()
    except Exception as exc:  # pragma: no cover - surfaced to the admin UI
        import traceback
        update_status(
            status='failed',
            message=f'Training failed: {exc}',
            finished_at=datetime.now().isoformat(timespec='seconds'),
            error=traceback.format_exc(),
        )


# ---------------------------------------------------------------------------
# Evaluation-only job (uses the existing h5 — no training)
# ---------------------------------------------------------------------------

def _run_evaluation(model_file: str = '') -> None:
    """Evaluate an h5 model on the test set.

    Much faster than training: it only does one forward pass over the test
    images and regenerates the bar chart, confusion matrix and pie chart.
    Training curves are skipped (they need training history).

    If ``model_file`` is provided (a filename like ``best_fer2013_vit.h5``),
    that file is loaded instead of the default ``best_mobilenetv2_fer2013.h5``.
    """
    update_status(
        status='running',
        message='Loading model and preparing the test set...',
        started_at=datetime.now().isoformat(timespec='seconds'),
        finished_at=None,
        plots=[],
        report='',
        metrics={},
        progress=10,
    )
    try:
        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix
        from tensorflow.keras.models import load_model
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
        except ImportError:
            from keras.preprocessing.image import ImageDataGenerator

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Determine which model file to load.
        if model_file:
            model_path = Path(settings.BASE_DIR) / model_file
        else:
            model_path = Path(settings.MODEL_PATH)
        if not model_path.is_file():
            raise FileNotFoundError(
                f'Model file not found at {model_path}. '
                'Train the model first ("Run Model") or place the h5 file in the project folder.'
            )
        update_status(status='running', message=f'Loading model: {model_path.name}...', progress=15)
        # Import vit_layers so custom layers (ClsToken, AddPosEmbed, etc.) are
        # registered before load_model tries to deserialize the h5 file.
        try:
            from . import vit_layers  # noqa: F401
        except ImportError:
            pass
        model = load_model(str(model_path), compile=False)

        # Pick the right preprocessing function for the model type.
        # Both MobileNetV2 and the custom ViT use [-1, 1] scaling, so
        # mobilenet_v2.preprocess_input works for both.
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        dataset_dir = Path(settings.BASE_DIR) / 'dataset'
        train_dir = dataset_dir / 'train'
        test_dir = dataset_dir / 'test'
        if not train_dir.is_dir() or not test_dir.is_dir():
            raise FileNotFoundError('Dataset folder not found: expected dataset/train and dataset/test.')

        classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        if not classes:
            raise FileNotFoundError('No emotion class folders found in dataset/train.')
        update_status(status='running', message=f'Found {len(classes)} emotion classes. Building test generator...', progress=25)

        train_counts = {}
        for cls in classes:
            train_counts[cls] = len([f for f in (train_dir / cls).iterdir() if f.is_file()])

        # Determine model input size and color mode from the loaded model.
        try:
            input_shape = tuple(model.inputs[0].shape)  # (None, H, W, C)
            model_h, model_w, model_c = int(input_shape[1]), int(input_shape[2]), int(input_shape[3])
        except Exception:
            model_h, model_w, model_c = 96, 96, 3
        color_mode = 'rgb' if model_c == 3 else 'grayscale'
        target_size = (model_h, model_w)

        # Use a small batch size for evaluation — large models (e.g. ViT 343 MB)
        # are very slow on CPU with big batches.
        eval_batch = 8
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_generator = test_datagen.flow_from_directory(
            str(test_dir), target_size=target_size, color_mode=color_mode,
            batch_size=eval_batch, class_mode='categorical', shuffle=False,
        )

        update_status(status='running', message='Running predictions on the test set...', progress=40)
        test_generator.reset()
        predictions = model.predict(test_generator, verbose=0)

        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        report_text = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        test_accuracy = float(np.mean(y_pred == y_true))

        # ---- Matplotlib graphs (bar chart, confusion matrix, pie chart) ----
        update_status(status='running', message='Generating graphs...', progress=80)
        out_dir = get_status_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        _plot_confusion_matrix(cm, classes, out_dir / 'confusion_matrix.png', plt, sns)
        _plot_class_distribution(train_counts, classes, test_dir, out_dir / 'class_distribution.png', plt, np)
        _plot_class_distribution_pie(train_counts, classes, out_dir / 'class_distribution_pie.png', plt)

        update_status(
            status='completed',
            message=f'Evaluation completed. Test accuracy: {test_accuracy:.4f} (from {model_path.name}, no training).',
            finished_at=datetime.now().isoformat(timespec='seconds'),
            progress=100,
            plots=[
                'training/confusion_matrix.png',
                'training/class_distribution.png',
                'training/class_distribution_pie.png',
            ],
            report=report_text,
            metrics={'test_accuracy': round(test_accuracy, 4), 'model_file': model_path.name},
        )
    except Exception as exc:  # pragma: no cover - surfaced to the admin UI
        import traceback
        update_status(
            status='failed',
            message=f'Evaluation failed: {exc}',
            finished_at=datetime.now().isoformat(timespec='seconds'),
            error=traceback.format_exc(),
        )


# ---------------------------------------------------------------------------
# Plot helpers (reproduce the notebook's figures)
# ---------------------------------------------------------------------------

def _plot_training_curves(history_stage1, history_stage2, path, plt):
    acc = list(history_stage1.history['accuracy']) + (list(history_stage2.history['accuracy']) if history_stage2 else [])
    val_acc = list(history_stage1.history['val_accuracy']) + (list(history_stage2.history['val_accuracy']) if history_stage2 else [])
    loss = list(history_stage1.history['loss']) + (list(history_stage2.history['loss']) if history_stage2 else [])
    val_loss = list(history_stage1.history['val_loss']) + (list(history_stage2.history['val_loss']) if history_stage2 else [])
    fine_tune_epoch = len(history_stage1.history['accuracy']) - 1

    fig = plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    if history_stage2:
        plt.axvline(x=fine_tune_epoch, color='red', linestyle='--', label='Fine-Tuning Start')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    if history_stage2:
        plt.axvline(x=fine_tune_epoch, color='red', linestyle='--', label='Fine-Tuning Start')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def _plot_confusion_matrix(cm, classes, path, plt, sns):
    fig = plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - MobileNetV2 on FER2013', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def _plot_class_distribution_pie(train_counts, classes, path, plt):
    """Pie chart of the training-set class share (like the notebook's Step 3)."""
    names = list(classes)
    counts = [train_counts.get(c, 0) for c in names]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _texts, autotexts = ax.pie(
        counts,
        labels=names,
        autopct=lambda p: f'{p:.1f}%' if p >= 1.5 else '',
        startangle=90,
        colors=plt.cm.tab10.colors[: len(names)],
        counterclock=False,
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title('FER2013 Training Class Distribution (Pie)')
    ax.axis('equal')
    plt.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def _plot_class_distribution(train_counts, classes, test_dir, path, plt, np):
    test_counts = {}
    for cls in classes:
        test_counts[cls] = len([f for f in (test_dir / cls).iterdir() if f.is_file()])

    names = list(classes)
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, [train_counts[c] for c in names], width, label='Train', color='#2f7de1')
    ax.bar(x + width / 2, [test_counts[c] for c in names], width, label='Test', color='#14b8a6')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel('Image Count')
    ax.set_title('FER2013 Class Distribution (Train vs Test)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
