import subprocess
import time
import urllib.request
import webbrowser
import sys
import os

def check_service(url):
    try:
        urllib.request.urlopen(url, timeout=1)
        return True
    except Exception:
        return False

def main():
    print("==================================================")
    print("        ThoughtGit Unified Startup Manager")
    print("==================================================")

    # 1. Check Ollama
    print(">>> Checking Ollama local service...")
    if check_service("http://localhost:11434"):
        print("✓ Ollama local service is ACTIVE.")
    else:
        print("⚠ WARNING: Ollama local service is OFFLINE.")
        print("Please make sure Ollama is started (or download it from https://ollama.com).")
        print("ThoughtGit will run in fallback mock-embedding mode until Ollama is active.\n")

    # 2. Find python executable in virtual env
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = sys.executable  # Fallback to current environment python

    print(f"Using python executable: {venv_python}")

    # 3. Start FastAPI Backend Server
    print(">>> Starting FastAPI API backend (port 8765)...")
    backend_proc = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8765"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for backend to activate
    backend_ready = False
    for _ in range(10):
        time.sleep(1)
        if check_service("http://127.0.0.1:8765/health"):
            backend_ready = True
            break

    if backend_ready:
        print("✓ FastAPI API backend is ACTIVE.")
    else:
        print("⚠ Backend server startup is taking longer than expected. Continuing...")

    # 4. Start Streamlit Dashboard
    print(">>> Starting Streamlit Dashboard (port 8501)...")
    dashboard_proc = subprocess.Popen(
        [venv_python, "-m", "streamlit", "run", "ui/dashboard.py", "--server.port", "8501"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait a moment for dashboard to start
    time.sleep(2)
    
    # 5. Open Web Browser
    dashboard_url = "http://localhost:8501"
    print(f"✓ Opening dashboard in browser: {dashboard_url}")
    webbrowser.open(dashboard_url)

    print("\nThoughtGit is running successfully!")
    print("Press Ctrl+C to terminate both servers and shutdown.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n>>> Terminating ThoughtGit servers...")
        backend_proc.terminate()
        dashboard_proc.terminate()
        backend_proc.wait()
        dashboard_proc.wait()
        print("Shutdown complete. Goodbye!")

if __name__ == "__main__":
    main()
