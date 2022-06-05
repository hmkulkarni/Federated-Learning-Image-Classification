from django import forms
from .models import *

# Create the form for choosing the image from MNIST dataset
class MnistForm(forms.ModelForm):
    class Meta:
        model = MnistImage
        fields = '__all__'
        labels = {'Image': ''}
        
        