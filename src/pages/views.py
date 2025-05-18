import os
import wave

from django import get_version
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST


def home(request):
    context = {
        "debug": settings.DEBUG,
        "django_ver": get_version(),
        "python_ver": os.environ["PYTHON_VERSION"],
    }

    return render(request, "pages/index.html", context)


@require_POST
def upload_wav(request):
    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return HttpResponseBadRequest("No file uploaded.")

    if not uploaded_file.name.lower().endswith(".wav"):
        return HttpResponseBadRequest("Invalid file type.")

    try:
        with wave.open(uploaded_file, "rb") as wav_file:
            wav_file.getparams()
    except BaseException:
        return HttpResponseBadRequest("Corrupted or non-wav file.")

    session_key = request.session.session_key
    if not session_key:
        request.session.create()
        session_key = request.session.session_key
    
    session_path = os.path.join(settings.MEDIA_ROOT, 'sessions', session_key)

    os.makedirs(session_path, exist_ok=True)
    fs = FileSystemStorage(location=session_path)
    filename = fs.save("upload.wav", uploaded_file)

    # TODO: start celery process

    return JsonResponse({'status': 'ok', 'filename': filename})