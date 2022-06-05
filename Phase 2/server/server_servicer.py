from fileTransfer_pb2 import DownloadOrder, UploadOrder, ServerMessage, DisconnectOrder
import fileTransfer_pb2_grpc
from fedavg import averageWeights
import torch
import os
import io

import sys
sys.path.insert(1, "../")

from client.train import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rounds = 0
completed_clients = 0
start, trained = False, False
class FileTransferServicer(fileTransfer_pb2_grpc.FileTransferServicer):
    def OpenChannel(self, request_iterator, context):
        """
        Responsible for file transfer to and from client
        """
        client_message_iterator = request_iterator
        global rounds, completed_clients, start, trained, device
        
        # Reset the no. of rounds to 0 if all clients disconnect (Since server doesn't stop)
        if completed_clients == 3:
            rounds = 0
            completed_clients = 0
            start = False
            trained = False
            
        if (not start) and rounds <= 2:
            print(f"\nStarting Communication Round {rounds + 1}")
            start = True
            
        # File names associated with communication rounds
        if rounds < 3:
            filename = "mnist.pt"
        elif rounds == 3:
            filename = "final_mnist.pt"
            
        # Read the file from the following path
        filepath = f"serverDB/CR{rounds}/" + filename
        with open(filepath, 'rb') as file:
            filedata = file.read()
        
        # Continue to train if less than 3 rounds are completed, else stop
        if rounds < 3:
            if not trained:
                print("\nTraining Clients")
                trained = True
            download_order_message = DownloadOrder(name = filename, data = filedata, ifTrain = 1)
        elif rounds == 3:
            if not trained:
                print("\nDone, Thank You!")
                trained = True
            download_order_message = DownloadOrder(name = filename, data = filedata, ifTrain = 0)
            completed_clients += 1
            
        # Give the training or stopping order to client
        message_to_client = ServerMessage(downloadOrder = download_order_message)
        yield message_to_client
        client_message = next(client_message_iterator)
        
        # Save the file received by the server in the respective communication round folder
        received_model = client_message.downloadAcknowledgement.data
        trained_model = torch.load(io.BytesIO(received_model))
        num = len(os.listdir(f"serverDB/CR{rounds + 1}/"))
        
        num_files = len(os.listdir(f'serverDB/CR{rounds + 1}/'))
        
        # If there are less than 3 files, then save the file and continue
        if num_files < 3:
            torch.save(trained_model, f"serverDB/CR{rounds + 1}/mnist{num + 1}.pt")
        
        num_files = len(os.listdir(f'serverDB/CR{rounds + 1}/'))
        
        # If there are 3 files in the CR folder, then perform federated average
        if num_files == 3:
            print("Performing Federated Average")
            m1 = torch.load(f"serverDB/CR{rounds + 1}/mnist1.pt")
            m2 = torch.load(f"serverDB/CR{rounds + 1}/mnist2.pt")
            m3 = torch.load(f"serverDB/CR{rounds + 1}/mnist3.pt")
            l = [m1, m2, m3]
        
            m = LeNetMnist().to(device)
            avg = averageWeights(l)
            if rounds < 2:
                torch.save(avg, f"serverDB/CR{rounds + 1}/mnist.pt")
                m.load_state_dict(torch.load(f"serverDB/CR{rounds + 1}/mnist.pt"))
            elif rounds == 2:
                torch.save(avg, f"serverDB/CR{rounds + 1}/final_mnist.pt")
                m.load_state_dict(torch.load(f"serverDB/CR{rounds + 1}/final_mnist.pt"))
            
            # Evaluate the testing accuracy of the model
            _, test_loader = loadDataset()
            test_acc = evaluation(m, test_loader)
            
            # Save the accuracy in the text file
            print(f"Testing accuracy of average model: {test_acc: 0.3f}%")
            with open("serverDB/averageWeights.txt", "a") as f:
                f.write(f"For Communication Round {rounds + 1}, Testing Accuracy - {test_acc: 0.3f}%\n")
            
            rounds += 1
            start = False
            trained = False

        # For disconnecting the client from the server
        disconnect_order_message = DisconnectOrder()
        yield disconnect_order_message
