from django.db import migrations


def create_admin_user(apps, schema_editor):
    User = apps.get_model('auth', 'User')
    if not User.objects.filter(username='admin').exists():
        User.objects.create_superuser(username='admin', email='admin@example.com', password='admin')


def remove_admin_user(apps, schema_editor):
    User = apps.get_model('auth', 'User')
    User.objects.filter(username='admin', is_superuser=True).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('emotion_app', '0001_initial'),
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.RunPython(create_admin_user, remove_admin_user),
    ]
