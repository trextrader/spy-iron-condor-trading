
import psutil
import os
import signal
import sys

def kill_all_python():
    current_pid = os.getpid()
    count = 0
    print(f"Scanning for Python processes (excluding self: {current_pid})...")
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if it's a python process
            if 'python' in proc.info['name'].lower() or \
               (proc.info['cmdline'] and 'python' in proc.info['cmdline'][0]):
                
                if proc.info['pid'] != current_pid:
                    print(f"Killing PID {proc.info['pid']}: {proc.info['cmdline']}")
                    os.kill(proc.info['pid'], signal.SIGKILL)
                    count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    print(f"Killed {count} background Python processes.")

if __name__ == "__main__":
    try:
        kill_all_python()
    except ImportError:
        print("psutil not installed. Attempting pkill fallback...")
        os.system("pkill -9 python")
