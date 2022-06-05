import torch
from train import *
import io

# Create a model from Architecture
mod = LeNetMnist().to(device)

def trainClient(model):
    """
    Trains the model
     
    Args: State-dict for a model
    Returns: State-dict of the trained model
    """
    mod.load_state_dict(torch.load(io.BytesIO(model)))
    train_loader, test_loader = loadDataset()
    train(mod, train_loader, test_loader)
    return mod.state_dict()