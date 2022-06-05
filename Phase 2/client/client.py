import grpc
from fileTransfer_pb2 import DownloadAcknowledgement, UploadAcknowledgement, ClientMessage
import fileTransfer_pb2_grpc
from queue import Queue
from clients_train import trainClient
import io
import torch
import os
from removefiles import removeFiles

# Clears all the existing files in the folder, for fresh start
removeFiles()

flag, comm_round, fin = 1, 1, 0
# Open the gRPC channel
with grpc.insecure_channel("localhost:8213") as channel:
    # Run the loop till stopping condition is met
    while flag:
        if fin == 0 and comm_round <= 3:
            print(f"\nCommunication Round {comm_round}\n")
            
        # Run the loop for 3 clients
        for num in range(1, 4):
            # Create a buffer for client (A queue for clients)
            stub = fileTransfer_pb2_grpc.FileTransferStub(channel)
            client_buffer = Queue(maxsize = 1)
            for server_message in stub.OpenChannel(iter(client_buffer.get, None)):
                
                if server_message.HasField("downloadOrder"):
                    # Receive the download order
                    download_order = server_message.downloadOrder
                    
                    # If "ifTrain == 0", then stop training and disconnect by breaking the loop
                    if download_order.ifTrain == 0:
                        print(f"\nSaving the final model in client {num}")
                        filename = download_order.name
                        filedata = download_order.data
                        filepath = f"clientDB/client{num}/{filename}"
                        with open(filepath, 'wb') as file:
                            file.write(filedata)
                        os.remove(f"clientDB/client{num}/mnist.pt")
                        fin += 1
                        print("Done")
                        if fin == 3:
                            flag = 0
                        break
                    
                    # Save the file in the desired path
                    filename = download_order.name
                    filedata = download_order.data
                    filepath = f"clientDB/client{num}/{filename}"
                    with open(filepath, 'wb') as file:
                        file.write(filedata)
                    print("Training client", num)
                    
                    # Run the code for training the client
                    name = trainClient(filedata)
                    
                    # Convert into bytes format in order for it to be parsable
                    buffer = io.BytesIO()
                    torch.save(name, buffer)
                    buffer.seek(0)
                    name_bytes = buffer.read()
                    
                    # Send the trained model as a message to server
                    message_to_server = ClientMessage(downloadAcknowledgement = DownloadAcknowledgement(data = name_bytes))
                    client_buffer.put(message_to_server)
                
                if server_message.HasField("disconnectOrder"):
                    #recieve server disconnect message
                    break
        
        comm_round += 1