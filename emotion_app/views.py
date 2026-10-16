import base64
import json
import logging
from functools import wraps
from pathlib import Path

from django.conf import settings
from django.contrib import messages as django_messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from .utils.train_model import (
    read_status as read_training_status,
    start_evaluation,
    start_training,
    scan_model_files,
)

logger = logging.getLogger(__name__)

from .forms import APISettingsForm, UserRegistrationForm
from .models import APISettings, EmotionRecord, UserProfile
from .utils.api_service import get_emotional_support_message
from .utils.emotion_model import NO_FACE_EMOTION, NO_FACE_LABEL, analyze_face_image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def admin_required(view_func):
    """Only superusers (the admin) may access the view."""
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_superuser:
            return redirect('admin_login')
        return view_func(request, *args, **kwargs)
    return _wrapped


# ---------------------------------------------------------------------------
# Public pages
# ---------------------------------------------------------------------------

def home(request):
    """Main project home page - project theme + overview."""
    return render(request, 'emotion_app/home.html')


# ---------------------------------------------------------------------------
# User registration / login
# ---------------------------------------------------------------------------

def user_register(request):
    if request.user.is_authenticated and not request.user.is_superuser:
        return redirect('user_home')
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            # New users are inactive until the admin approves them.
            user.is_active = False
            user.save()
            UserProfile.objects.create(user=user, phone=form.cleaned_data.get('phone', ''))
            django_messages.success(
                request,
                'Registration successful! Your account is pending admin approval. '
                'Please wait until the admin activates your account, then log in.',
            )
            return redirect('user_login')
    else:
        form = UserRegistrationForm()
    return render(request, 'emotion_app/user_register.html', {'form': form})


def user_login(request):
    if request.user.is_authenticated:
        if request.user.is_superuser:
            return redirect('admin_home')
        return redirect('user_home')

    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = User.objects.filter(username=username).first()
        if user is not None and user.check_password(password):
            if user.is_active:
                login(request, user)
                return redirect('user_home')
            error = 'Your account is not activated yet. Please wait for the admin to approve your registration.'
        else:
            error = 'Invalid username or password.'
    return render(request, 'emotion_app/user_login.html', {'error': error})


def user_logout(request):
    logout(request)
    return redirect('home')


# ---------------------------------------------------------------------------
# User dashboard
# ---------------------------------------------------------------------------

@login_required
@ensure_csrf_cookie
def user_home(request):
    """User dashboard. ``ensure_csrf_cookie`` guarantees the csrftoken cookie
    exists so the camera JS can POST to /analyze/ without a 403."""
    if request.user.is_superuser:
        return redirect('admin_home')
    records = EmotionRecord.objects.filter(user=request.user)[:12]
    return render(request, 'emotion_app/user_home.html', {'records': records})


# ---------------------------------------------------------------------------
# Emotion analysis endpoint (camera capture / image upload)
# ---------------------------------------------------------------------------

@require_POST
def analyze(request):
    """Accept an image (file upload or base64 from the webcam) and return
    { emotion, confidence, message }."""
    try:
        return _analyze(request)
    except Exception as exc:
        logger.exception('analyze endpoint crashed')
        return JsonResponse({'success': False, 'error': f'Server error: {exc}'})


def _analyze(request):
    """Inner implementation of the analyze endpoint."""
    # Only logged-in users reach the scanner page; don't let anonymous
    # requests run the 328 MB model.
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Please log in to analyze images.'})

    img_bytes = None
    source = 'camera'

    if request.FILES.get('image'):
        uploaded = request.FILES['image']
        if uploaded.size > MAX_UPLOAD_BYTES:
            return JsonResponse({'success': False,
                                 'error': 'The image is too large (max 10 MB). Please choose a smaller image.'})
        img_bytes = uploaded.read()
        source = 'upload'
    else:
        try:
            payload = json.loads(request.body or b'{}')
        except json.JSONDecodeError:
            payload = {}
        data_url = payload.get('image', '')
        if not data_url:
            return JsonResponse({'success': False, 'error': 'No image received.'})
        if data_url.startswith('data:'):
            data_url = data_url.split(',', 1)[1]
        try:
            img_bytes = base64.b64decode(data_url)
        except Exception:
            return JsonResponse({'success': False, 'error': 'Could not decode the image.'})

    result = analyze_face_image(img_bytes)
    if result.get('no_face'):
        # Invalid input (mobile, plant, chart ...): report it as its own result
        # and keep it in the history instead of inventing an emotion.
        message = ('The uploaded image does not contain a human face. '
                   'Please upload or capture a clear, front-facing photo of a face.')
        _save_record(request, NO_FACE_EMOTION, 0.0, message, source, img_bytes)
        return JsonResponse({
            'success': False,
            'no_face': True,
            'label': NO_FACE_LABEL,
            'error': NO_FACE_LABEL,
            'message': message,
        })
    if result.get('error'):
        return JsonResponse({'success': False, 'error': result['error']})

    emotion = result['emotion']
    confidence = result['confidence']
    message = get_emotional_support_message(emotion)

    # History keeps the labeled image so every face's prediction can be seen later.
    annotated = result.get('annotated_image')
    _save_record(request, emotion, confidence, message, source, img_bytes, annotated)

    return JsonResponse({
        'success': True,
        'emotion': emotion,
        'confidence': round(confidence * 100, 1),
        'message': message,
        'main_face_id': result.get('main_face_id'),
        'faces': [
            {'id': f['id'], 'emotion': f['emotion'], 'confidence': round(f['confidence'] * 100, 1)}
            for f in result.get('faces', [])
        ],
        'annotated_image': ('data:image/jpeg;base64,' + base64.b64encode(annotated).decode('ascii')
                            if annotated else None),
    })


def _save_record(request, emotion, confidence, message, source, img_bytes, annotated=None):
    """Store a history record. Saving must never hide the detection result."""
    if not request.user.is_authenticated:
        return
    try:
        record = EmotionRecord(
            user=request.user,
            emotion=emotion,
            confidence=confidence,
            message=message,
            source=source,
        )
        if annotated:
            record.image.save(f'{source}_labeled.jpg', ContentFile(annotated), save=False)
        elif source == 'upload' and request.FILES.get('image'):
            uploaded = request.FILES['image']
            uploaded.seek(0)
            record.image.save(uploaded.name, uploaded, save=False)
        else:
            record.image.save('capture.jpg', ContentFile(img_bytes), save=False)
        record.save()
    except Exception:
        logger.exception('Failed to save detection history record')


# ---------------------------------------------------------------------------
# Admin authentication
# ---------------------------------------------------------------------------

def admin_login(request):
    if request.user.is_authenticated and request.user.is_superuser:
        return redirect('admin_home')

    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = User.objects.filter(username=username).first()
        if user is not None and user.check_password(password):
            if user.is_superuser:
                login(request, user)
                return redirect('admin_home')
            error = 'This account is not an admin account.'
        else:
            error = 'Invalid admin username or password.'
    return render(request, 'emotion_app/admin_login.html', {'error': error})


def admin_logout(request):
    logout(request)
    return redirect('admin_login')


# ---------------------------------------------------------------------------
# Admin dashboard
# ---------------------------------------------------------------------------

def _dataset_stats():
    """Count images per emotion class for train/test splits."""
    base = Path(settings.BASE_DIR) / 'dataset'
    stats = {}
    total = {'train': 0, 'test': 0}
    if base.exists():
        for split in ('train', 'test'):
            split_dir = base / split
            if not split_dir.is_dir():
                continue
            for cls_dir in sorted(split_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                count = sum(
                    1 for f in cls_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                )
                stats.setdefault(cls_dir.name, {})[split] = count
                total[split] += count
    return stats, total


@admin_required
def admin_home(request):
    stats, total = _dataset_stats()
    users = User.objects.all().order_by('-date_joined')
    records_total = EmotionRecord.objects.count()

    context = {
        'users': users,
        'total_users': users.count(),
        'active_users': users.filter(is_active=True).count(),
        'inactive_users': users.filter(is_active=False).count(),
        'records_total': records_total,
        'dataset_stats': stats,
        'dataset_total': total,
        'api_settings': APISettings.get_settings(),
        'api_form': APISettingsForm(instance=APISettings.get_settings()),
        'active_tab': request.GET.get('tab', 'dashboard'),
        'training_status': read_training_status(),
        'model_exists': settings.MODEL_PATH.exists(),
        'inference_model_exists': getattr(settings, 'INFERENCE_MODEL_PATH', settings.MODEL_PATH).exists(),
        'inference_model_name': getattr(settings, 'INFERENCE_MODEL_PATH', settings.MODEL_PATH).name,
        'model_files': scan_model_files(),
        'current_model': settings.MODEL_PATH.name,
    }
    return render(request, 'emotion_app/admin_home.html', context)


@admin_required
@require_POST
def admin_train_model(request):
    """Start retraining the emotion model (per fer2013_mobilenetv2.ipynb).

    Training runs in a background thread so the request returns instantly;
    progress is polled via ``admin_training_status``.
    """
    model_file = request.POST.get('model_file', '').strip()
    ok, message = start_training(model_file=model_file)
    if ok:
        django_messages.success(request, message)
    else:
        django_messages.error(request, message)
    return redirect(reverse('admin_home') + '?tab=training')


@admin_required
@require_POST
def admin_evaluate_model(request):
    """Evaluate the existing h5 model on the test set (no training)."""
    model_file = request.POST.get('model_file', '').strip()
    ok, message = start_evaluation(model_file=model_file)
    if ok:
        django_messages.success(request, message)
    else:
        django_messages.error(request, message)
    return redirect(reverse('admin_home') + '?tab=training')


@admin_required
def admin_training_status(request):
    """JSON status + results used by the admin panel to poll the training job."""
    return JsonResponse(read_training_status())


@admin_required
@require_POST
def admin_user_action(request):
    """Activate / deactivate / delete a user from the admin panel."""
    user_id = request.POST.get('user_id')
    action = request.POST.get('action')
    user = User.objects.filter(pk=user_id).first()

    if user is None:
        django_messages.error(request, 'User not found.')
    elif user.is_superuser:
        django_messages.error(request, 'You cannot modify the admin account.')
    elif action == 'activate':
        user.is_active = True
        user.save(update_fields=['is_active'])
        django_messages.success(request, f'User "{user.username}" has been activated.')
    elif action == 'deactivate':
        user.is_active = False
        user.save(update_fields=['is_active'])
        django_messages.success(request, f'User "{user.username}" has been deactivated.')
    elif action == 'delete':
        name = user.username
        user.delete()
        django_messages.success(request, f'User "{name}" has been deleted.')
    else:
        django_messages.error(request, 'Unknown action.')

    return redirect('admin_home')


@admin_required
@require_POST
def admin_save_api(request):
    """Save the user-provided API key (Gemini / ChatGPT) settings."""
    settings_obj = APISettings.get_settings()
    form = APISettingsForm(request.POST, instance=settings_obj)
    if form.is_valid():
        form.save()
        django_messages.success(request, 'API settings saved successfully.')
    else:
        django_messages.error(request, 'Please fix the errors in the form.')
    return redirect('admin_home')
