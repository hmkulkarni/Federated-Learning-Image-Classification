from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from .models import *
from .forms import *
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import urllib
from PIL import Image
import cv2
import os
from pathlib import Path
from .mnist_lenet import *
import torch.nn.functional as F
import subprocess

# Create your views here.
def home(request):
    """
    Returns the homepage
    
    Args: request
    Returns: View of the respective url provided
    """
    return render(request, 'home.html')

def mnist(request):
    """
    Returns the page for choosing the MNIST image
    
    Args: request
    Returns: View of the respective url provided, along with the form for choosing image
    """
    if request.method == "POST":
        
        # Initialize an object of MnistForm mentioned in forms.py file
        mnistForm = MnistForm(request.POST, request.FILES)
        
        # If the form is valid, then save the form and proceed, or else redisplay the form
        if mnistForm.is_valid():
            mnistForm.save()
        return redirect("mnist_result")
    mnistForm = MnistForm()
    return render(request, 'mnist/mnist.html', {'form': mnistForm})

def mnistResult(request):
    """
    Does the inferencing on the average model after applying
    federated learning.
    
    Args: request
    Returns: View of the respective url provided, along with image, predicted value
            and probabilities
    """
    img = MnistImage.objects.latest('id') # Get id of image from database
    
    # Set the image path to media folder in local storage
    base = Path(__file__).resolve().parent.parent
    mediaPath = os.path.join(base, 'media')
    imgPath = os.path.join(mediaPath, img.Image.path)

    # Save the name of image in text file
    imgName = img.Image.path.split("\\")[-1]
    with open("savedImage.txt", "a") as f:
        f.write(f"Image name: {imgName}\t")
    
    # Open the subprocess for client training in federated learning
    # and make the parent process (the current function) wait
    p = subprocess.Popen("python classify/grpc_transfer/client/client.py", shell=True)
    p.wait()
    
    # If child process is complete, then do the inferencing on averaged model
    if p.returncode == 0:
        model = torch.load("classify/grpc_transfer/server/serverDB/CR3/final_mnist.pt")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Read the image, and resize it to 32*32 image since LeNet model requires
        # dimensions as 32*32 whereas the images in MNIST dataset are of 28*28
        image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image)
        image = image.float()
        image = image.to(device)
        
        # Add another dimension to match with the dimensions of required image in the model
        image = image.unsqueeze(0)
        
        # Predict the label of the image
        predicted = predictMnist(model, image)
        
        # Find the top k probabilities along with the labels
        top_p, top_class = predicted.topk(10, dim = 1)
        
        # Merge these 2 lists in order to easily iterate through them in frontend 
        probabilities = zip(top_class[0].tolist(), top_p[0].tolist())
        
        # Save the predicted label in the variable
        pred_val = predicted[0].argmax(0).item()

        # Save the label of the corresponding image id in the text file mentioned above
        with open("savedImage.txt", "a") as f:
            f.write(f"Predicted Label: {pred_val}\n")
            
        return render(request, 'mnist/mnist_result.html', 
                    {'img': img, 'predicted': pred_val, 
                    'probs': probabilities})
    
    # If some error occurs during completion of child process, then show internal server error
    else:
        return HttpResponse(status=500)