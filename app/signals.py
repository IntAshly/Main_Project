from django.contrib.auth.signals import user_logged_out
from django.dispatch import receiver
from .models import Notification

@receiver(user_logged_out)
def clear_notifications(sender, request, user, **kwargs):
    # Clear all vaccine reminder notifications for the user
    Notification.objects.filter(user=user, message__icontains='Reminder:').delete()

    # Clear vaccine reminder session data
    reminder_keys = [key for key in request.session.keys() if key.startswith('vaccine_reminder_')]
    for key in reminder_keys:
        del request.session[key]