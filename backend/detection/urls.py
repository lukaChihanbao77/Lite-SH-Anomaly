"""
检测模块URL配置
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from detection.views import DetectionViewSet

router = DefaultRouter()
router.register('', DetectionViewSet, basename='detection')

urlpatterns = [
    path('', include(router.urls)),
]
