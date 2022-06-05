import torch
import torch.nn as nn
import torch.nn.functional as F

# Creating LeNet architecture for MNIST
class LeNetMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # First Convolutional Layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # First Pooling Layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Second Convulational Layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Second Pooling Layer
        self.fc1 = nn.Linear(400, 120)  # First Fully Connected Layer (Input size is 16 kernels, each of 5*5 so 16*5*5 = 400)
        self.fc2 = nn.Linear(120, 84)  # Second Fully Connected Layer
        self.fc3 = nn.Linear(84, 10)  # Third Fully Connected Layer
        
    def forward(self, x):
        x = F.relu(self.conv1(x))  # Applying ReLU function after first convolutional layer
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))  # Applying ReLU function after second convolutional layer
        x = self.pool2(x)
        
        x = x.view(-1, 400)  # Flattening it for passing it in fully connected layer (Size => 16 kernels, 5*5 each => 16*5*5 = 400)
        
        x = F.relu(self.fc1(x))  # Applying ReLU function after first fully connected layer
        x = F.relu(self.fc2(x))  # Applying ReLU function after second fully connected layer
        x = self.fc3(x)  # Applying ReLU function after third fully connected layer
        
        return F.log_softmax(x, -1)
    
def predictMnist(model, image):
    """
    Predicts the labels given the image from MNIST dataset

    Args: Model and image to be classified
    Returns: Probabilities of the labels
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = LeNetMnist().to(device)
    m.load_state_dict(model)
    m.eval()
    with torch.no_grad():
        pred = m(image.unsqueeze(0))
        probs = F.softmax(pred, dim=1)
        preds = torch.argmax(pred, dim=1)
    return probs