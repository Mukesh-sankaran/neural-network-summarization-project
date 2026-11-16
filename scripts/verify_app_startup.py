"""
Bounded Streamlit Startup and Health Verification Script.
Starts the Streamlit application as a background subprocess, polls the health endpoint
until HTTP 200 OK is received within a strict timeout, and cleanly terminates the process.
Ensures that no hanging processes or port leaks occur.
"""

import os
import sys
import time
import subprocess
import urllib.request
import signal

PORT = 8501
URL = f"http://localhost:{PORT}/_stcore/health"
TIMEOUT_SECS = 20


def main():
    print(f"Launching Streamlit application on port {PORT} with {TIMEOUT_SECS}s bounded timeout...")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    start_time = time.time()
    healthy = False

    try:
        while time.time() - start_time < TIMEOUT_SECS:
            # Check if process terminated prematurely
            ret = proc.poll()
            if ret is not None:
                stderr = proc.stderr.read()
                print(f"[ERROR] Streamlit process exited prematurely with code {ret}:\n{stderr}")
                sys.exit(1)

            try:
                with urllib.request.urlopen(URL, timeout=1.0) as resp:
                    if resp.status == 200:
                        content = resp.read().decode("utf-8").strip()
                        print(f"[PASS] Streamlit health endpoint responded in {time.time() - start_time:.2f}s (Status 200: '{content}').")
                        healthy = True
                        break
            except Exception:
                time.sleep(0.5)

        if not healthy:
            print(f"[FAIL] Streamlit server failed to respond at {URL} within {TIMEOUT_SECS} seconds.")
            sys.exit(1)

    finally:
        print("Gracefully shutting down Streamlit server process...")
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        print("[PASS] Streamlit application shutdown clean and verified.")


if __name__ == "__main__":
    main()
