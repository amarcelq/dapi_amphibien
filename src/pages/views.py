import os
import wave

from django import get_version
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST, require_GET


def home(request):
    return render(request, "pages/index.html")


@require_POST
def upload_wav(request):
    uploaded_file = request.FILES.get("file")

    session_key = request.session.session_key
    if not session_key:
        return HttpResponseBadRequest(
            "Session key is missing. A session should always exist."
        )

    if not uploaded_file:
        return HttpResponseBadRequest("No file uploaded.")

    if not uploaded_file.name.lower().endswith(".wav"):
        return HttpResponseBadRequest("Invalid file type.")

    try:
        with wave.open(uploaded_file, "rb") as wav_file:
            wav_file.getparams()
    except BaseException:
        return HttpResponseBadRequest("Corrupted or non-wav file.")

    session_path = os.path.join(settings.MEDIA_ROOT, "sessions", session_key)

    os.makedirs(session_path, exist_ok=True)
    filename = os.path.join("sessions", session_key, "upload.wav")
    file_path = os.path.join(settings.MEDIA_ROOT, filename)
    # TODO: not good? may be different based on used FileSystem
    if default_storage.exists(file_path):
        default_storage.delete(file_path)

    filename = default_storage.save(filename, uploaded_file)

    # TODO: start celery process

    return JsonResponse({"status": "ok", "filename": filename})


@require_GET
def result(request):
    session_key = request.session.session_key
    if not session_key:
        return HttpResponseBadRequest(
            "Session key is missing. A session should always exist."
        )

    res = {
        "main_audio": {"name": "Original", "url": "/media/sessions/p8j9rvbb3b135e4cgr2hfd46bnufs7uq/upload.wav"}, # the main, original uploaded audio track from the user
        "alternative_audio": [{"name": "Denoised", "url": ""}], # alternative tracks that were created during processing (e.g. a denoised Version). They should all have the same length and SampleRate
        # All found unique samples
        "samples": [
            {
                "id": "id-to-be-implemented", # session-unique id for this sample type
                "name": "#1", # Each sample has a displayed name
                "snippets": [{"url": "", "start": 1000, "end": 2000}], # and snippets, which each point to a unique snippet url. The start and end time in ms is relative to the original audio track
            },
            {
                "id": "id-to-be-implemented", # session-unique id for this sample type
                "name": "#2", # Each sample has a displayed name
                "snippets": [{"url": "", "start": 3000, "end": 3500}], # and snippets, which each point to a unique snippet url. The start and end time in ms is relative to the original audio track
            }
        ],
    }

    return JsonResponse(res)
