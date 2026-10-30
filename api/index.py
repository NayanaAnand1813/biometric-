import os
# Prevent OpenCV threading deadlocks in serverless environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from pathlib import Path
from urllib.parse import parse_qs

# Add project root to sys.path so app and its modules are discoverable
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from app import app

# WSGI Middleware to restore original request path from Vercel's rewrite parameter
class VercelPathFixMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        query_string = environ.get("QUERY_STRING", "")
        if "__path=" in query_string:
            params = parse_qs(query_string, keep_blank_values=True)
            if "__path" in params and params["__path"]:
                raw_path = params["__path"][0]
                target_path = "/" + raw_path.lstrip("/")
                environ["PATH_INFO"] = target_path
                # Clean __path from QUERY_STRING so it doesn't pollute request.args
                clean_qs = "&".join(
                    f"{k}={v}" for k, vals in params.items() if k != "__path" for v in vals
                )
                environ["QUERY_STRING"] = clean_qs
            else:
                environ["PATH_INFO"] = "/"
        elif environ.get("PATH_INFO") in ("/api/index", "/api/index/"):
            environ["PATH_INFO"] = "/"
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
