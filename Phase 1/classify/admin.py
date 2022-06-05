from django.contrib import admin
from .models import *

# Register your models here.
@admin.register(MnistImage)
class MnistImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'Image']