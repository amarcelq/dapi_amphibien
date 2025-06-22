from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from importlib import import_module
from django.conf import settings
from django.contrib.sessions.backends.base import SessionBase
import os
from django.http import HttpResponseBadRequest, FileResponse, Http404
from django.conf import settings
from zipfile import ZipFile
import tempfile

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

@csrf_exempt
def progress_finish(request):
    if request.method == "POST":
        data = json.loads(request.body)

        session_key = data.get("session_key")
        progress = data.get("result")
        session:SessionBase = SessionStore(session_key=session_key)
        session["result"] = progress #{"status":"done","name":"Finished","description":""}
        print("Saving: ", progress)
        session.save()
        return JsonResponse({"status": "ok"})
    return JsonResponse({"error": "invalid method"}, status=405)

@csrf_exempt
def download_session_folder(request):
    session_key = request.session.session_key
    if not session_key:
        return HttpResponseBadRequest(
            "Session key is missing. A session should always exist."
        )
    folder_path = os.path.join(settings.MEDIA_ROOT, 'sessions', session_key)
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise Http404("Session folder not found.")
    
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with ZipFile(temp, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname=rel_path)
    
    temp.seek(0)
    response = FileResponse(temp, as_attachment=True, filename=f"{session_key}.zip")
    return response