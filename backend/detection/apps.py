"""
检测应用配置
"""

from django.apps import AppConfig


class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'
    verbose_name = '异常检测'
    
    def ready(self):
        """应用启动时加载模型"""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            from detection.services.model_service import model_service
            loaded = model_service.load_model()
            if loaded:
                logger.info('检测模型加载成功')
            else:
                logger.warning('检测模型未加载，使用模拟模式')
        except Exception as e:
            logger.error(f'模型加载失败: {e}')
