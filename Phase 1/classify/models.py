from django.db import models

# Create the model for MNIST images(For storing in django-handled database)
class MnistImage(models.Model):
    Image = models.ImageField(upload_to='mnist/')
    
    def __unicode__(self):
        return "MNIST"