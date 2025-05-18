import os
import shutil

from django.conf import settings
from django.contrib.sessions.models import Session
from django.core.management.base import BaseCommand
from django.utils.timezone import now


class Command(BaseCommand):
    def handle(self, *args, **options):
        session_dir = os.path.join(settings.MEDIA_ROOT, 'sessions')
        if not os.path.isdir(session_dir):
            return

        active_sessions = set(
            Session.objects.filter(expire_date__gt=now()).values_list('session_key', flat=True)
        )

        for dirname in os.listdir(session_dir):
            path = os.path.join(session_dir, dirname)
            if os.path.isdir(path) and dirname not in active_sessions:
                self.stdout.write(f"Removed {path}")
                shutil.rmtree(path)
