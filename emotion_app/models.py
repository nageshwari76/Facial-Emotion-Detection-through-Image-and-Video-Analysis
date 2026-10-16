from django.conf import settings
from django.db import models


class UserProfile(models.Model):
    """Extra profile information for a registered user."""
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='profile',
    )
    phone = models.CharField(max_length=15, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.username


class EmotionRecord(models.Model):
    """A single emotion detection performed by a user."""
    SOURCE_CHOICES = [
        ('camera', 'Camera'),
        ('upload', 'Image Upload'),
    ]
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='emotion_records',
    )
    image = models.ImageField(upload_to='emotion_captures/', blank=True, null=True)
    emotion = models.CharField(max_length=20)
    confidence = models.FloatField(default=0.0)
    message = models.TextField(blank=True)
    source = models.CharField(max_length=10, choices=SOURCE_CHOICES, default='camera')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user_id or "guest"} - {self.emotion} - {self.created_at:%Y-%m-%d %H:%M}'


class APISettings(models.Model):
    """Blank API configuration so the owner can plug in their own key
    (Google Gemini or OpenAI ChatGPT) for personalized messages."""
    PROVIDER_CHOICES = [
        ('none', 'Disabled'),
        ('gemini', 'Google Gemini'),
        ('openai', 'OpenAI ChatGPT'),
    ]
    provider = models.CharField(max_length=20, choices=PROVIDER_CHOICES, default='none')
    api_key = models.CharField(max_length=255, blank=True, help_text='Paste your own Gemini / ChatGPT API key here')
    model_name = models.CharField(max_length=100, blank=True, help_text='e.g. gemini-2.0-flash or gpt-4o-mini (optional)')
    enabled = models.BooleanField(default=False, help_text='Tick to start using the API for messages')
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = 'API Settings'

    @classmethod
    def get_settings(cls):
        obj, _created = cls.objects.get_or_create(pk=1)
        return obj

    def __str__(self):
        return self.provider
