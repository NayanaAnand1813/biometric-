import sys
from pathlib import Path

# Add project root to sys.path so app and its modules are discoverable
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from app import app

# WSGI handler for Vercel Serverless Function

