from concurrent import futures

import grpc
import fileTransfer_pb2
import fileTransfer_pb2_grpc

from server_servicer import FileTransferServicer

def server_start():
    """
    Starts the gRPC server for communication between client and server
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FileTransferServicer()
    fileTransfer_pb2_grpc.add_FileTransferServicer_to_server(servicer, server)
    server.add_insecure_port('localhost:8213')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    server_start()