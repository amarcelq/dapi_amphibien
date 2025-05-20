from django.urls import path

from pages import views

urlpatterns = [
    path("", views.home, name="home"),
    path("wav", views.upload_wav, name="upload_wav"),
    path("result", views.result, name="result")
]
