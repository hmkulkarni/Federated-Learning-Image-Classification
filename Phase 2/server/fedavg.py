import torch
import copy

def averageWeights(w):
    """
    Computes the federated average of the weights.
    
    Args: List of multiple state dicts
    Returns: State dict of averaged model
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg