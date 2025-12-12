"""
WebSocket消费者
实现实时检测推送
"""

import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)


class DetectionConsumer(AsyncWebsocketConsumer):
    """实时检测WebSocket消费者"""
    
    GROUP_NAME = 'detection_realtime'
    
    async def connect(self):
        """WebSocket连接建立"""
        await self.channel_layer.group_add(
            self.GROUP_NAME,
            self.channel_name
        )
        await self.accept()
        
        # 发送连接成功消息
        await self.send(text_data=json.dumps({
            'type': 'connection',
            'status': 'connected',
            'message': '实时检测连接已建立'
        }))
        
        logger.info(f'WebSocket连接建立: {self.channel_name}')
    
    async def disconnect(self, close_code):
        """WebSocket连接断开"""
        await self.channel_layer.group_discard(
            self.GROUP_NAME,
            self.channel_name
        )
        logger.info(f'WebSocket连接断开: {self.channel_name}, code={close_code}')
    
    async def receive(self, text_data):
        """接收客户端消息"""
        try:
            data = json.loads(text_data)
            action = data.get('action')
            
            if action == 'ping':
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'message': 'pong'
                }))
            
            elif action == 'detect':
                # 实时检测请求
                detection_data = data.get('data', {})
                result = await self.perform_detection(detection_data)
                await self.send(text_data=json.dumps({
                    'type': 'detection_result',
                    'data': result
                }))
            
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': '无效的JSON格式'
            }))
        except Exception as e:
            logger.error(f'WebSocket处理错误: {e}')
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def detection_alert(self, event):
        """接收检测告警并推送给客户端"""
        await self.send(text_data=json.dumps({
            'type': 'alert',
            'data': event['data']
        }))
    
    async def detection_result(self, event):
        """接收检测结果并推送给客户端"""
        await self.send(text_data=json.dumps({
            'type': 'detection_result',
            'data': event['data']
        }))
    
    @database_sync_to_async
    def perform_detection(self, data):
        """执行检测（同步转异步）"""
        from detection.services import DetectionService
        return DetectionService.detect_single(data)


class RealtimePusher:
    """实时推送工具类"""
    
    @staticmethod
    async def push_detection_result(result: dict):
        """推送检测结果到所有连接的客户端"""
        from channels.layers import get_channel_layer
        channel_layer = get_channel_layer()
        
        await channel_layer.group_send(
            DetectionConsumer.GROUP_NAME,
            {
                'type': 'detection_result',
                'data': result
            }
        )
    
    @staticmethod
    async def push_alert(alert_data: dict):
        """推送告警到所有连接的客户端"""
        from channels.layers import get_channel_layer
        channel_layer = get_channel_layer()
        
        await channel_layer.group_send(
            DetectionConsumer.GROUP_NAME,
            {
                'type': 'detection_alert',
                'data': alert_data
            }
        )
