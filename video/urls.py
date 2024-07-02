from django.urls import path
from .views import *

urlpatterns = [
    path('', index),

        # 'livefe' -> function from views
        # 'live_camera' -> name at index.html>img src="{% url 'live_camera' %}
        path('camera', livefe, name="live_camera"),
    ]