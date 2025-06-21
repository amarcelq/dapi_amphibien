# inseart app specifc tasks here
from celery import shared_task

from importlib import import_module
from django.conf import settings
from django.contrib.sessions.backends.base import SessionBase
SessionStore:SessionBase = import_module(settings.SESSION_ENGINE).SessionStore
from pathlib import Path
import requests

@shared_task
def process(session_key):
    session:SessionBase = SessionStore(session_key=session_key)
    session["status"] = {"status":"running","name":"Process Started","description":"The process was started."}
    session.save()
    upload = session["original_upload_location"]

    requests.post("audio:7777/start_process",json={"session_key":session_key, "path":upload})
    return session_key