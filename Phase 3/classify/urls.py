from django.urls import path
from . import views

# Sets the URL endpoints on the webpage
urlpatterns = [
    path('', views.home, name='home'),
    path('mnist/', views.mnist, name='mnist'),
    path('mnist_result/', views.mnistResult, name='mnist_result'),
]