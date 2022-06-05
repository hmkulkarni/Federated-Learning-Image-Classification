import os

def removeFiles():
    """
    Removes the files in the client and server model directories if any
    """
    for num in range(1, 4):
        dir = f"../client/clientDB/client{num}/"
        for file in os.scandir(dir):
            os.remove(file.path)
            
    for num in range(1, 4):
        dir = f"../server/serverDB/CR{num}/"
        for file in os.scandir(dir):
            os.remove(file.path)