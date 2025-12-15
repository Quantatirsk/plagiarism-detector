#!/usr/bin/env python
"""
FastAPI 服务启动脚本
使用方法: cd backend && python run.py
"""
import os
import platform
import signal
import socket
import subprocess
import time

import uvicorn


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return False
        except OSError:
            return True


def kill_process_on_port(port: int, force: bool = False) -> bool:
    try:
        system = platform.system()
        if system in ("Darwin", "Linux"):
            cmd = f"lsof -t -i:{port}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout.strip():
                for pid in result.stdout.strip().split('\n'):
                    try:
                        os.kill(int(pid), signal.SIGKILL if force else signal.SIGTERM)
                        print(f"Terminated process {pid}")
                    except Exception:
                        continue
                return True
        elif system == "Windows":
            cmd = f"netstat -ano | findstr :{port}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if "LISTENING" in line:
                        parts = line.split()
                        if len(parts) > 4:
                            subprocess.run(f"taskkill /F /PID {parts[-1]}", shell=True)
                return True
    except Exception:
        pass
    return False


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    if is_port_in_use(port):
        print(f"Port {port} in use, killing...")
        kill_process_on_port(port, force=False)
        time.sleep(1)
        if is_port_in_use(port):
            kill_process_on_port(port, force=True)
            time.sleep(1)

    print("=" * 50)
    print(f"Server: http://{host}:{port}")
    print(f"Docs: http://localhost:{port}/docs")
    print("=" * 50)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
