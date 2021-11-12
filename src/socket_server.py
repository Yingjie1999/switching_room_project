import socket
import traceback

import primary_main

# address = ('0.0.0.0', 5005)  # 服务端地址和端口
address = ('127.0.0.1', 5005)  # 服务端地址和端口
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(address)  # 绑定服务端地址和端口
s.listen(5)
conn, addr = s.accept()  # 返回客户端地址和一个新的 socket 连接
print('[+] Connected with', addr)
while True:
    data = conn.recv(1024)  # buffersize 等于 1024
    data = data.decode()
    if not data:
        break
    print('[Received]', data)
    if data == ("ready"):
        try:
            primary_main.total_recogntion()
        except Exception as e:
            traceback.print_exc()
    # send = input('Input: ')
    send = ('OK')
    conn.sendall(send.encode())
conn.close()
s.close()