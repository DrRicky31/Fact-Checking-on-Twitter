import socket
import time
import os
import random

data_path = '../data/russian-troll-tweets/'

# Creazione di un socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 9999))
server_socket.listen(1)
print("Server in ascolto sulla porta 9999...")

connection, address = server_socket.accept()
print("Connessione stabilita con:", address)

filelist = os.listdir(data_path)
filelist_subset = random.sample(filelist, 2)

for file in filelist_subset:
    with open(data_path + file, 'r') as f:
        for line in f:
            connection.sendall(line.encode('utf-8'))
            time.sleep(0.1)  # Ritardo per simulare lo streaming

connection.close()
server_socket.close()
