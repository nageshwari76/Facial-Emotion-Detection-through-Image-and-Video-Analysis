from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.user_register, name='user_register'),
    path('login/', views.user_login, name='user_login'),
    path('logout/', views.user_logout, name='user_logout'),
    path('user-home/', views.user_home, name='user_home'),
    path('analyze/', views.analyze, name='analyze'),
    path('admin-login/', views.admin_login, name='admin_login'),
    path('admin-home/', views.admin_home, name='admin_home'),
    path('admin-user-action/', views.admin_user_action, name='admin_user_action'),
    path('admin-save-api/', views.admin_save_api, name='admin_save_api'),
    path('admin-train-model/', views.admin_train_model, name='admin_train_model'),
    path('admin-evaluate-model/', views.admin_evaluate_model, name='admin_evaluate_model'),
    path('admin-training-status/', views.admin_training_status, name='admin_training_status'),
    path('admin-logout/', views.admin_logout, name='admin_logout'),
]
