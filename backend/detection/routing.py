"""
WebSocket路由配置
"""

from django.urls import path
from detection.consumers import DetectionConsumer

websocket_urlpatterns = [
    path('ws/detection/', DetectionConsumer.as_asgi()),
]
