"""API integration for personalized emotional support messages.

The user can plug in their own API key (Google Gemini or OpenAI ChatGPT)
from the admin panel. If no key is configured, a built-in fallback message
is shown instead.
"""
import random

import requests

from ..models import APISettings

FALLBACK_MESSAGES = {
    'angry': [
        'It looks like you are feeling angry. Take a deep breath — count to ten, step away for a minute, and let the wave pass. Your calm is stronger than the storm.',
        'Anger is a signal, not a verdict. Try some slow breathing and a short walk to release the tension.',
    ],
    'disgust': [
        'You seem disgusted by something right now. A change of scene can help reset your mood — step outside or listen to music you love.',
        'When something feels off, give yourself permission to move on. Focus on the things that bring you comfort.',
    ],
    'fear': [
        'You seem a little fearful. Remember: you have handled difficult moments before. Take slow, deep breaths and focus on what you can control.',
        'Fear is temporary. Name what worries you, then do one small, calming action — a warm drink, a walk, or talking to someone you trust.',
    ],
    'happy': [
        'You look happy! Savor this moment — happiness grows when we share it. Tell someone about what is making you smile today.',
        'Great to see you smiling! Channel this positive energy into something you enjoy.',
    ],
    'neutral': [
        'You seem neutral right now. It is okay to just be. Maybe try a tiny mood booster — your favourite song or a short stretch.',
        'A calm, steady state is a great place to be. Use this time to focus on something you have been wanting to do.',
    ],
    'sad': [
        'It looks like you are feeling low. Be kind to yourself — sadness is part of being human. Reach out to a friend or do something gentle that comforts you.',
        'You are not alone in feeling this way. A warm cup of tea, a favourite movie, or a quiet walk can make a real difference today.',
    ],
    'surprise': [
        'Something surprising caught your attention! Enjoy the moment of wonder — curiosity keeps life exciting.',
        'Surprise is a spark of energy. Take a moment to appreciate the unexpected and let it lift your mood.',
    ],
}

DEFAULT_FALLBACK = [
    'Thank you for checking in with yourself. Take a deep breath and remember: your feelings are valid, and you are doing better than you think.',
]


def build_prompt(emotion):
    return (
        f'The detected emotion of the user is "{emotion}". '
        'Please respond in 4 short lines with: (1) a personalized motivational message, '
        '(2) a line of emotional support, (3) a stress-relief suggestion, and '
        '(4) one recommended activity. Keep it warm, friendly and concise.'
    )


# Short timeout so an offline/unreachable network fails fast and the app
# falls back to the built-in messages instead of hanging.
API_TIMEOUT = 8


def call_gemini(settings_obj, prompt):
    model = settings_obj.model_name.strip() or 'gemini-2.0-flash'
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={settings_obj.api_key.strip()}'
    payload = {'contents': [{'parts': [{'text': prompt}]}]}
    resp = requests.post(url, json=payload, timeout=API_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data['candidates'][0]['content']['parts'][0]['text'].strip()


def call_openai(settings_obj, prompt):
    model = settings_obj.model_name.strip() or 'gpt-4o-mini'
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Authorization': f'Bearer {settings_obj.api_key.strip()}'}
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data['choices'][0]['message']['content'].strip()


def get_emotional_support_message(emotion):
    """Return a personalized message for the emotion.

    Uses the configured API (Gemini / ChatGPT) when enabled, otherwise a
    built-in fallback message.
    """
    settings_obj = APISettings.get_settings()
    if settings_obj.enabled and settings_obj.provider in ('gemini', 'openai') and settings_obj.api_key.strip():
        try:
            prompt = build_prompt(emotion)
            if settings_obj.provider == 'gemini':
                return call_gemini(settings_obj, prompt)
            return call_openai(settings_obj, prompt)
        except Exception:
            # Fall through to the built-in message if the API call fails.
            pass

    options = FALLBACK_MESSAGES.get(emotion, DEFAULT_FALLBACK)
    return random.choice(options)
