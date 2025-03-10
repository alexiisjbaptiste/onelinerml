import subprocess
import time
import sys
from pyngrok import ngrok

def start_servers():
    API_PORT = 8000
    STREAMLIT_PORT = 8501

    api_process = subprocess.Popen(
        ["uvicorn", "onelinerml.api:app", "--host", "0.0.0.0", "--port", str(API_PORT)]
    )
    time.sleep(2)

    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "onelinerml/dashboard.py"]
    )
    time.sleep(2)

    api_tunnel = ngrok.connect(API_PORT)
    streamlit_tunnel = ngrok.connect(STREAMLIT_PORT)

    print("FastAPI is running at:", api_tunnel.public_url)
    print("Streamlit Dashboard is running at:", streamlit_tunnel.public_url)

    return api_process, streamlit_process

def main():
    api_proc, streamlit_proc = start_servers()
    print("\nPress Ctrl+C to shut down the servers...\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down servers and tunnels...")
        api_proc.terminate()
        streamlit_proc.terminate()
        ngrok.kill()
        sys.exit(0)

if __name__ == "__main__":
    main()
