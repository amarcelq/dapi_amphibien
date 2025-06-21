from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from importlib import import_module
from django.conf import settings
from django.contrib.sessions.backends.base import SessionBase
SessionStore:SessionBase = import_module(settings.SESSION_ENGINE).SessionStore

@csrf_exempt
def progress_update(request):
    if request.method == "POST":
        data = json.loads(request.body)

        session_key = data.get("session_key")
        progress = data.get("progress")
        session:SessionBase = SessionStore(session_key=session_key)
        session["status"] = progress #{"status":"done","name":"Finished","description":""}
        session.save()
        return JsonResponse({"status": "ok"})
    return JsonResponse({"error": "invalid method"}, status=405)