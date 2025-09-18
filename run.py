#!/usr/bin/env python
"""
快捷启动脚本 - 直接运行 FastAPI 应用
使用方法: python run.py 或 ./run.py
支持自动清理端口占用
"""
import sys
import os
import socket
import signal
import subprocess
import platform
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))


def is_port_in_use(port: int) -> bool:
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return False
        except OSError:
            return True


def get_process_info_on_port(port: int) -> str:
    """获取占用端口的进程信息"""
    try:
        system = platform.system()
        
        if system == "Darwin" or system == "Linux":
            # 获取进程信息
            cmd = f"lsof -i:{port} | grep LISTEN"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if lines:
                    # 解析进程信息
                    parts = lines[0].split()
                    if len(parts) > 1:
                        return f"Process: {parts[0]} (PID: {parts[1]})"
        elif system == "Windows":
            cmd = f"netstat -ano | findstr :{port}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "LISTENING" in line:
                        parts = line.split()
                        if len(parts) > 4:
                            return f"PID: {parts[-1]}"
    except:
        pass
    
    return "Unknown process"


def kill_process_on_port(port: int, force: bool = False) -> bool:
    """终止占用指定端口的进程"""
    try:
        system = platform.system()
        
        if system == "Darwin" or system == "Linux":
            # macOS 和 Linux
            cmd = f"lsof -t -i:{port}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        # 使用 SIGTERM (优雅关闭) 或 SIGKILL (强制关闭)
                        sig = signal.SIGKILL if force else signal.SIGTERM
                        os.kill(int(pid), sig)
                        print(f"✅ Terminated process {pid} using port {port}")
                    except:
                        continue
                return True
                
        elif system == "Windows":
            # Windows
            cmd = f"netstat -ano | findstr :{port}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                pids_killed = set()
                for line in lines:
                    if "LISTENING" in line:
                        parts = line.split()
                        if len(parts) > 4:
                            pid = parts[-1]
                            if pid not in pids_killed:
                                force_flag = "/F" if force else ""
                                subprocess.run(f"taskkill {force_flag} /PID {pid}", shell=True)
                                pids_killed.add(pid)
                                print(f"✅ Terminated process {pid} using port {port}")
                return len(pids_killed) > 0
                
    except Exception as e:
        print(f"⚠️ Error killing process on port {port}: {e}")
    
    return False


# 主程序
if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量获取配置
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    auto_clean = os.getenv("AUTO_CLEAN", "true").lower() == "true"
    
    # 检查端口是否被占用
    if is_port_in_use(port):
        process_info = get_process_info_on_port(port)
        print(f"⚠️ Port {port} is already in use by {process_info}")
        
        if auto_clean:
            print(f"🔧 AUTO_CLEAN is enabled. Attempting to free port {port}...")
            
            # 尝试优雅关闭
            if kill_process_on_port(port, force=False):
                time.sleep(2)  # 等待端口释放
                
                # 如果还被占用，强制关闭
                if is_port_in_use(port):
                    print(f"⚠️ Port {port} still in use. Force killing...")
                    kill_process_on_port(port, force=True)
                    time.sleep(1)
            
            if is_port_in_use(port):
                print(f"❌ Failed to free port {port}. Please manually terminate the process.")
                print(f"You can also use a different port: PORT=8001 python {__file__}")
                sys.exit(1)
        else:
            # 询问用户
            try:
                response = input(f"Do you want to terminate the process using port {port}? (y/n): ").lower()
                if response == 'y':
                    if kill_process_on_port(port):
                        time.sleep(2)
                        if is_port_in_use(port):
                            print("⚠️ Port still in use. Force killing...")
                            kill_process_on_port(port, force=True)
                            time.sleep(1)
                    
                    if is_port_in_use(port):
                        print(f"❌ Failed to free port {port}")
                        sys.exit(1)
                else:
                    print(f"Use a different port: PORT=8001 python {__file__}")
                    print(f"Or enable auto-clean: AUTO_CLEAN=true python {__file__}")
                    sys.exit(1)
            except KeyboardInterrupt:
                print("\n👋 Cancelled")
                sys.exit(0)
    
    # 启动配置
    print("=" * 60)
    print("🚀 FastAPI 向量嵌入文档检测系统")
    print("=" * 60)
    print(f"📍 Server: http://{host}:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"🔧 Interactive API: http://localhost:{port}/redoc")
    print("=" * 60)
    print("Press CTRL+C to stop the server")
    print()
    
    try:
        # 启动服务器
        uvicorn.run(
            "backend.main:app",  # 使用字符串导入以支持 reload
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        sys.exit(0)