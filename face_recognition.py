"""Face Recognition System - Main Application Launcher.

Launches the Flask biometric web application with silent terminal mode
and automatically opens the 2-Option Portal (Register Face / Face Login)
in the browser.
"""

from app import run_app

if __name__ == "__main__":
    run_app()
