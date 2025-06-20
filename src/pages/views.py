import os
import wave

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpResponseBadRequest, JsonResponse
from django.http.request import HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from pages.tasks import process


def home(request:HttpRequest):
    request.session['init'] = True
    request.session.save()
    return render(request, "pages/index.html")


@require_POST
def upload_wav(request:HttpRequest):
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
    request.session["original_upload_location"] =  os.path.join(settings.MEDIA_ROOT, filename)
    request.session["status"] = {"status":"pending","name":"Waiting for Worker","description":"The Task is pending and will be accepted by the next free worker."}
    request.session.save()
    # TODO: start celery process
    process.delay(session_key)

    return JsonResponse({"status": "ok", "filename": filename})


@require_GET
def result(request):
    session_key = request.session.session_key
    if not session_key:
        return HttpResponseBadRequest(
            "Session key is missing. A session should always exist."
        )

    # for now its a semi-static response, until sound backend does stuff
    res = {
        "main_audio": {"name": "Original", "url": f"/media/sessions/{session_key}/upload.wav"}, # the main, original uploaded audio track from the user
        "alternative_audio": [{"name": "Denoised", "url": ""}], # alternative tracks that were created during processing (e.g. a denoised Version). They should all have the same length and SampleRate
        # All found unique samples
        "samples": [
            {
                "id": "id-to-be-implemented", # session-unique id for this sample type
                "name": "#1", # Each sample has a displayed name
                "snippets": [{"url": f"/media/sessions/{session_key}/cut.wav", "start": 10000, "duration": 5000}], # and snippets, which each point to a unique snippet url. The start and end time in ms is relative to the original audio track. The first snippet will be displayed in big
            },
            # {
            #     "id": "id-to-be-implemented", # session-unique id for this sample type
            #     "name": "#2", # Each sample has a displayed name
            #     "snippets": [{"url": "", "start": 3500, "duration": 1000}], # and snippets, which each point to a unique snippet url. The start and end time in ms is relative to the original audio track
            # }
        ],
    }

    return JsonResponse(res)


@require_GET
def progress(request: HttpRequest):
    session_key = request.session.session_key
    if not session_key:
        return HttpResponseBadRequest(
            "Session key is missing. A session should always exist."
        )

    status = request.session.get("status")
    if status is None:
        return JsonResponse({"status": "unknown", "name":"","description":""})

    return JsonResponse(status)