"""
自定义异常类
"""

from rest_framework.exceptions import APIException
from rest_framework import status


class ServiceException(APIException):
    """业务逻辑异常"""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = '业务处理失败'
    default_code = 'service_error'


class ModelNotLoadedException(ServiceException):
    """模型未加载异常"""
    default_detail = '检测模型未加载'
    default_code = 'model_not_loaded'


class InvalidDataException(ServiceException):
    """无效数据异常"""
    default_detail = '数据格式无效'
    default_code = 'invalid_data'


class DeviceNotFoundException(ServiceException):
    """设备不存在异常"""
    status_code = status.HTTP_404_NOT_FOUND
    default_detail = '设备不存在'
    default_code = 'device_not_found'
