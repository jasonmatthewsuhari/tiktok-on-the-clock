#!/usr/bin/env python3
"""
Enhanced server startup script for TikTok Data Processing Pipeline API.
Runs uvicorn with proper configuration and error handling.
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path
import webbrowser
from threading import Timer

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI and Uvicorn are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install required packages:")
        print("pip install fastapi uvicorn")
        return False

def check_app_file():
    """Check if app.py exists"""
    if not Path("app.py").exists():
        print("❌ app.py not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        print("Please make sure you're in the correct project directory")
        return False
    print("✅ app.py found")
    return True

def check_virtual_env():
    """Check if virtual environment is activated"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is activated")
        return True
    else:
        print("⚠️  Virtual environment not detected")
        print("Consider activating your virtual environment:")
        print("venv\\Scripts\\activate  # Windows")
        print("source venv/bin/activate  # Linux/Mac")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'data', 'config']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created/verified directory: {dir_name}")

def open_browser_delayed():
    """Open browser after a delay"""
    time.sleep(3)  # Wait 3 seconds for server to start
    try:
        webbrowser.open("http://localhost:8000")
        print("🌐 Opened browser at http://localhost:8000")
    except Exception as e:
        print(f"Could not open browser: {e}")

def start_server():
    """Start the FastAPI server with uvicorn"""
    
    print("🚀 TikTok Data Processing Pipeline API")
    print("=" * 50)
    
    # Pre-flight checks
    if not check_app_file():
        return False
    
    if not check_dependencies():
        return False
    
    check_virtual_env()
    create_directories()
    
    print("\n🔧 Server Configuration:")
    print("  Host: 0.0.0.0 (all interfaces)")
    print("  Port: 8000")
    print("  Reload: Enabled (auto-restart on code changes)")
    print("  URL: http://localhost:8000")
    
    print("\n📚 Available Endpoints:")
    print("  🏠 Root: http://localhost:8000/")
    print("  🚀 Start Pipeline: POST http://localhost:8000/pipeline/start")
    print("  📊 Pipeline Status: GET http://localhost:8000/pipeline/status/{id}")
    print("  📋 Execution List: GET http://localhost:8000/pipeline/executions")
    print("  📄 Logs: GET http://localhost:8000/pipeline/logs/{id}")
    print("  ❤️  Health Check: GET http://localhost:8000/health")
    
    print("\n" + "=" * 50)
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Optional: Open browser automatically
    browser_timer = Timer(3.0, open_browser_delayed)
    browser_timer.daemon = True
    browser_timer.start()
    
    try:
        # Start the server
        result = subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload",
            "--log-level", "info"
        ], check=False)
        
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        return True
    except FileNotFoundError:
        print("❌ uvicorn not found. Please install it:")
        print("pip install uvicorn")
        return False
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

def main():
    """Main function"""
    success = start_server()
    
    if not success:
        print("\n💥 Server failed to start")
        input("Press Enter to exit...")
        sys.exit(1)
    else:
        print("\n✅ Server stopped successfully")

if __name__ == "__main__":
    main()
