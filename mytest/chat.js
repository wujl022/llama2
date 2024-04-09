// 创建 WebSocket 连接
const socket = io();

// 监听来自服务器的消息
socket.on('message', function(data) {
    console.log('Received message:', data);
});

// 发送消息给服务器
socket.emit('message', 'Hello, server!');
