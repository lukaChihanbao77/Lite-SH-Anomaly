"""
URL configuration for Lite-SH-Anomaly project.
RESTful API路由配置
"""

from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView
)

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # API文档
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
    
    # API接口
    path('api/v1/', include([
        path('devices/', include('devices.urls')),
        path('detection/', include('detection.urls')),
        path('logs/', include('logs.urls')),
    ])),
]
