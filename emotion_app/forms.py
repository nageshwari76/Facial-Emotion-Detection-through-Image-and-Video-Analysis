from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import APISettings


class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email address'}))
    phone = forms.CharField(max_length=15, required=False, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Phone (optional)'}))

    class Meta:
        model = User
        fields = ['username', 'email', 'phone', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Choose a username'})
        self.fields['password1'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Password'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Confirm password'})

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError('An account with this email already exists.')
        return email


class APISettingsForm(forms.ModelForm):
    class Meta:
        model = APISettings
        fields = ['provider', 'api_key', 'model_name', 'enabled']
        widgets = {
            'provider': forms.Select(attrs={'class': 'form-control'}),
            'api_key': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Paste your API key here (e.g. AIza..., sk-...)'}),
            'model_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Optional model name'}),
            'enabled': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
