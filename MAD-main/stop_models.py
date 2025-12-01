#!/usr/bin/env python3
"""
Stop all running litgpt models
"""
import subprocess
import signal
import os
import time

def stop_models():
    """Stop all running litgpt models"""
    print("🛑 Stopping all litgpt models...")
    
    # Find all litgpt processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        litgpt_pids = []
        for line in lines:
            if 'litgpt serve' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    litgpt_pids.append(pid)
                    print(f"   Found litgpt process: PID {pid}")
        
        if not litgpt_pids:
            print("✅ No litgpt processes found")
            return True
        
        # Kill each process
        for pid in litgpt_pids:
            try:
                print(f"   Killing process {pid}...")
                os.kill(int(pid), signal.SIGTERM)
                time.sleep(1)
                
                # Check if process is still running
                try:
                    os.kill(int(pid), 0)  # Check if process exists
                    print(f"   Process {pid} still running, using SIGKILL...")
                    os.kill(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    print(f"   ✅ Process {pid} stopped successfully")
                    
            except ProcessLookupError:
                print(f"   ✅ Process {pid} already stopped")
            except Exception as e:
                print(f"   ❌ Error stopping process {pid}: {e}")
        
        # Wait a moment and verify
        time.sleep(2)
        
        # Check if any processes are still running
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        remaining_processes = []
        for line in result.stdout.split('\n'):
            if 'litgpt serve' in line and 'grep' not in line:
                remaining_processes.append(line)
        
        if remaining_processes:
            print("⚠️ Some processes may still be running:")
            for proc in remaining_processes:
                print(f"   {proc}")
        else:
            print("✅ All litgpt models stopped successfully!")
            
    except Exception as e:
        print(f"❌ Error stopping models: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("=" * 60)
    print("🛑 LITGPT MODEL STOPPER")
    print("=" * 60)
    
    success = stop_models()
    
    print("=" * 60)
    if success:
        print("🎉 All models stopped successfully!")
    else:
        print("⚠️ Some issues occurred while stopping models")
    print("=" * 60)

if __name__ == "__main__":
    main()
