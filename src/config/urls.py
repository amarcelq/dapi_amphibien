"""
URL configuration for frogs project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from django.conf.urls.static import static
import config.views as views

urlpatterns = [
    path("up/", include("up.urls")),
    path("", include("pages.urls")),
    path("internal/progress/update/", views.progress_update),
    path("internal/progress/finish/", views.progress_finish)
    # path("admin/", admin.site.urls),
]

if not settings.TESTING:
    urlpatterns = [
        *urlpatterns,
        path("__debug__/", include("debug_toolbar.urls")),
    ]

# Media files should be served via a different service/url, for multiple reasons. 
# In development just add it to the url, so its available.
# So, unless we want to setup nginx, S3 or sth like that, just stay in debug mode
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)