"""
统一响应格式
"""

from rest_framework.response import Response
from rest_framework import status


class APIResponse:
    """统一API响应格式"""
    
    @staticmethod
    def success(data=None, message='success', code=200):
        """成功响应"""
        return Response({
            'code': code,
            'message': message,
            'data': data
        }, status=status.HTTP_200_OK)
    
    @staticmethod
    def created(data=None, message='创建成功'):
        """创建成功响应"""
        return Response({
            'code': 201,
            'message': message,
            'data': data
        }, status=status.HTTP_201_CREATED)
    
    @staticmethod
    def error(message='操作失败', code=400, data=None):
        """错误响应"""
        return Response({
            'code': code,
            'message': message,
            'data': data
        }, status=status.HTTP_400_BAD_REQUEST)
    
    @staticmethod
    def not_found(message='资源不存在'):
        """404响应"""
        return Response({
            'code': 404,
            'message': message,
            'data': None
        }, status=status.HTTP_404_NOT_FOUND)
    
    @staticmethod
    def paginated(queryset, serializer_class, request):
        """分页响应"""
        from rest_framework.pagination import PageNumberPagination
        paginator = PageNumberPagination()
        page = paginator.paginate_queryset(queryset, request)
        if page is not None:
            serializer = serializer_class(page, many=True)
            return Response({
                'code': 200,
                'message': 'success',
                'data': {
                    'count': paginator.page.paginator.count,
                    'next': paginator.get_next_link(),
                    'previous': paginator.get_previous_link(),
                    'results': serializer.data
                }
            })
        serializer = serializer_class(queryset, many=True)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })
