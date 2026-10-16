from django.contrib import admin

from .models import APISettings, EmotionRecord, UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone', 'created_at')
    search_fields = ('user__username', 'phone')


@admin.register(EmotionRecord)
class EmotionRecordAdmin(admin.ModelAdmin):
    list_display = ('user', 'emotion', 'confidence', 'source', 'created_at')
    list_filter = ('emotion', 'source', 'created_at')
    search_fields = ('user__username', 'emotion')


@admin.register(APISettings)
class APISettingsAdmin(admin.ModelAdmin):
    list_display = ('provider', 'enabled', 'updated_at')
