"""
WSGI config for ImgClassification project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os
import subprocess
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ImgClassification.settings')

subprocess.Popen("python classify/grpc_transfer/server/server.py", shell=True)

application = get_wsgi_application()
