from django.http import HttpResponseForbidden


class InternalOnlyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith("/internal/"):
            ip = request.META.get("REMOTE_ADDR")
            if not ip.startswith("172.") and ip != "127.0.0.1":
                return HttpResponseForbidden("Forbidden")
        return self.get_response(request)