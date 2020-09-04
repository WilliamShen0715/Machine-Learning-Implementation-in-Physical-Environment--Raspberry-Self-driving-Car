import socket
tcpServerSocket=socket.socket()

host = '192.168.43.88'
port=12345
tcpServerSocket.bind((host,port))
tcpServerSocket.listen(5)
c, addr = tcpServerSocket.accept() 
c_msg = c.recv(4096).decode()
print(c_msg)
while True:
	say = input("輸入你想傳送的訊息：")
	c.send(say.encode())
