from django import template

register = template.Library()

EMOTION_EMOJI = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲',
    'no_face': '🚫',
}


@register.filter
def emotion_emoji(value):
    return EMOTION_EMOJI.get(str(value).lower(), '😶')


@register.filter
def emotion_label(value):
    """Display name for a stored emotion ('no_face' -> 'No face image detected')."""
    if str(value).lower() == 'no_face':
        return 'No face image detected'
    return str(value).title()
