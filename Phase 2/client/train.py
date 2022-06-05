import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# Set default device as cuda if cuda is installed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadDataset():
    """
    Downloads the dataset for MNIST and creates a data-loader
    
    Args: None
    Returns: Data loaders for train and test data
    """
    
    transformations = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    batch_size = 50
    train_dataset = torchvision.datasets.MNIST(
        root="./MNIST",  # Folder where the images are to be stored
        train=True,  # Specifies that the training data is to be taken
        transform=transformations,  # Converts the image data to tensor
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./MNIST",  # Folder where the images are to be stored
        train=False,  # Specifies that the training data is to be taken
        transform=transformations,  # Converts the image data to tensor
        download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,   # Good for training since it reorders the dataset for better training
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,   # No need to shuffle for evaluation
    )
    return train_loader, test_loader

# Create a LeNet architecture for MNIST
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


def evaluation(model, dataloader):
    """
    Evaluates the training and testing accuracy of the given model
    
    Args: model file, train or test dataloader
    Returns: Training or Testing accuracy depending on data loader
    """
    total , correct = 0, 0
    for images, labels in dataloader:
        # Loads the image and label from data loader to device
        images, labels = images.to(device), labels.to(device)
        
        # Pass the image to the model
        output = model(images)
        
        # Get the predicted output
        _, pred = torch.max(output.data, dim=1)
        total += labels.shape[0]
        correct += (pred == labels).sum().item()
    return (100 * correct) / total


def train(model, train_loader, test_loader, max_epochs=1):
    criterion = nn.NLLLoss()  # Negative Log Likelihood Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_arr = []
    
    for epoch in range(max_epochs):
        # Iterating through the train loader
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

             # Forward Pass
            outputs = model(images)  # Store the output
            loss = criterion(outputs, labels)  # Computes the loss

            # Backward Pass
            optimizer.zero_grad() # Clears the old gradients that are stored while computing
            loss.backward()  # Does backpropagation
            optimizer.step() # Updates the weights (Gradient descent)

            # Print info about loss
            if (i + 1) % 1200 == 0:
                print(f"Epoch: {epoch + 1}/{max_epochs}, Step: {i + 1}/{len(train_loader)}, Loss: {loss.item(): 0.4f}")
            
            # To keep track of losses (if required)
            loss_arr.append(loss.item())
        
        train_acc = evaluation(model, train_loader)
        test_acc = evaluation(model, test_loader)
        
        print(f"Training Accuracy: {train_acc: 0.3f}%, Testing Accuracy: {test_acc: 0.3f}%\n")