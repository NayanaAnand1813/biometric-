import os
# Prevent OpenCV threading deadlocks in serverless environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from pathlib import Path

# Add project root to sys.path so app and its modules are discoverable
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from app import app

# WSGI Middleware to restore original request path from Vercel's x-matched-path
class VercelPathFixMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        matched_path = environ.get("HTTP_X_MATCHED_PATH")
        if matched_path:
            environ["PATH_INFO"] = matched_path
        elif environ.get("PATH_INFO") in ("/api/index", "/api/index/"):
            environ["PATH_INFO"] = "/"
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
