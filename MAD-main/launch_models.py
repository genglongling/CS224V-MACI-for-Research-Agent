#!/usr/bin/env python3
"""
Launch script for three Qwen models using litgpt
"""
import subprocess
import time
import signal
import sys
from pathlib import Path

def launch_model(checkpoint_path: str, port: int, model_name: str):
    """Launch a single model on the specified port"""
    cmd = [
        "litgpt", "serve", checkpoint_path, 
        "--port", str(port),
        "--max_new_tokens", "16384"  # Increase from default 50 to 16K
        # Note: litgpt serve defaults to 0.0.0.0 (all interfaces)
    ]
    
    print(f"🚀 Launching {model_name} on port {port}...")
    print(f"Command: {' '.join(cmd)}")
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint path does not exist: {checkpoint_path}")
        print(f"   Please download the model to: {checkpoint_path}")
        return None
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Wait longer for the model to start and check output
        print(f"⏳ Waiting for {model_name} to start on port {port}...")
        
        # Check process status and output for first 30 seconds
        for i in range(30):
            time.sleep(1)
            
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"❌ {model_name} failed to start on port {port}")
                print(f"Exit code: {process.returncode}")
                print(f"stdout: {stdout}")
                print(f"stderr: {stderr}")
                return None
            
            # Check if port is listening
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                if result == 0:
                    print(f"✅ {model_name} started successfully on port {port} (PID: {process.pid})")
                    return process
            except:
                pass
            
            if i % 5 == 0:
                print(f"   Still starting... ({i+1}/30 seconds)")
        
        # If we get here, the process is running but port might not be ready
        print(f"⚠️ {model_name} process is running but port {port} might not be ready yet")
        print(f"   Process PID: {process.pid}")
        return process
            
    except Exception as e:
        print(f"❌ Error launching {model_name} on port {port}: {e}")
        return None

def check_litgpt():
    """Check if litgpt is available"""
    try:
        # First check if the command exists
        result = subprocess.run(["which", "litgpt"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("❌ litgpt command not found in PATH")
            print("   Install with: pip install litgpt")
            return False
        
        print("✅ litgpt command found, testing functionality...")
        
        # Try to run litgpt help to check if it's available
        result = subprocess.run(["litgpt", "--help"], 
                              capture_output=True, text=True, timeout=30)  # Increased timeout
        if result.returncode == 0:
            print("✅ litgpt is available and working")
            return True
        else:
            print(f"❌ litgpt command failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ litgpt command not found. Please install litgpt first.")
        print("   Install with: pip install litgpt")
        return False
    except subprocess.TimeoutExpired:
        print("❌ litgpt command timed out (took longer than 30 seconds)")
        print("   This might indicate an issue with the litgpt installation")
        print("   Try: pip uninstall litgpt && pip install litgpt")
        return False
    except Exception as e:
        print(f"❌ Error checking litgpt: {e}")
        return False

def main():
    """Launch all three models"""
    print("🎯 Launching three Qwen models for MAD benchmark...")
    
    # Show current working directory
    current_dir = Path.cwd()
    print(f"📁 Current working directory: {current_dir}")
    
    # Check if litgpt is available
    if not check_litgpt():
        print("❌ Cannot proceed without litgpt. Please install it first.")
        return False
    
    # Check checkpoint directory
    checkpoint_dir = Path("checkpoints/Qwen/Qwen2.5-7B-Instruct")
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        print("   Please download the Qwen2.5-7B-Instruct model first.")
        print("   Expected structure:")
        print(f"   {checkpoint_dir}/")
        print("   ├── lit_model.pth")
        print("   ├── tokenizer.json")
        print("   └── ...")
        return False
    
    # Check for essential files
    essential_files = ["lit_model.pth", "tokenizer.json"]
    missing_files = []
    for file in essential_files:
        if not (checkpoint_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing essential files in {checkpoint_dir}:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print(f"✅ Checkpoint directory found: {checkpoint_dir}")
    
    # Test models function
    def test_models():
        """Test if all models are responding"""
        import requests
        import time
        
        print("   Testing model endpoints...")
        ports = [8000, 8001, 8003]
        names = ["Agent A", "Agent B", "Judge"]
        
        all_working = True
        for port, name in zip(ports, names):
            try:
                # Try different endpoints that litgpt might have
                endpoints = ["/health", "/", "/v1/models"]
                working = False
                
                for endpoint in endpoints:
                    try:
                        response = requests.get(f"http://localhost:{port}{endpoint}", timeout=5)
                        if response.status_code == 200:
                            print(f"     ✅ {name} (port {port}) - {endpoint}")
                            working = True
                            break
                    except:
                        continue
                
                if not working:
                    print(f"     ❌ {name} (port {port}) - no working endpoint found")
                    all_working = False
                    
            except Exception as e:
                print(f"     ❌ {name} (port {port}) - error: {e}")
                all_working = False
        
        return all_working
    
    # Model configurations
    models = [
        {
            "checkpoint": "checkpoints/Qwen/Qwen2.5-7B-Instruct",
            "port": 8000,
            "name": "Agent A"
        },
        {
            "checkpoint": "checkpoints/Qwen/Qwen2.5-7B-Instruct", 
            "port": 8001,
            "name": "Agent B"
        },
        {
            "checkpoint": "checkpoints/Qwen/Qwen2.5-7B-Instruct",
            "port": 8003,
            "name": "Judge"
        }
    ]
    
    processes = []
    
    try:
        # Launch each model
        for model in models:
            process = launch_model(
                model["checkpoint"], 
                model["port"], 
                model["name"]
            )
            if process:
                processes.append((process, model["name"], model["port"]))
            else:
                print(f"❌ Failed to launch {model['name']}, stopping...")
                return False
        
        if len(processes) == 3:
            print(f"\n🎉 All three models launched successfully!")
            print(f"   Agent A: http://localhost:8000")
            print(f"   Agent B: http://localhost:8001") 
            print(f"   Judge:   http://localhost:8003")
            
            # Test the models to make sure they're working
            print(f"\n🧪 Testing models...")
            if test_models():
                print(f"✅ All models are working correctly!")
            else:
                print(f"⚠️ Some models may not be working properly")
            
            print(f"\nPress Ctrl+C to stop all models...")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
                    # Check if any process has died
                    for process, name, port in processes:
                        if process.poll() is not None:
                            print(f"⚠️ {name} on port {port} has stopped unexpectedly")
                            print(f"   Exit code: {process.returncode}")
                            # Try to get any output
                            try:
                                stdout, stderr = process.communicate(timeout=1)
                                if stdout: print(f"   stdout: {stdout}")
                                if stderr: print(f"   stderr: {stderr}")
                            except:
                                pass
            except KeyboardInterrupt:
                print(f"\n🛑 Stopping all models...")
                
        else:
            print(f"❌ Only {len(processes)}/3 models started successfully")
            return False
            
    except KeyboardInterrupt:
        print(f"\n🛑 Stopping all models...")
    finally:
        # Stop all processes
        for process, name, port in processes:
            if process.poll() is None:
                print(f"🛑 Stopping {name} on port {port}...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                    print(f"✅ {name} stopped")
                except subprocess.TimeoutExpired:
                    print(f"⚠️ Force killing {name}...")
                    process.kill()
                    process.wait()
        
        print("✅ All models stopped")

if __name__ == "__main__":
    main()
