import time
import torch
import numpy as np
import os
import sys

# Add path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BoardEvaluator import HexNet
from utils import HexPlanes

def measure_latency(device_name, model, input_tensor, num_warmup=50, num_runs=1000):
    device = torch.device(device_name)
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
            
    # Synchronize for GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def test_feature_generation_latency(num_runs=1000):
    board = np.zeros((11, 11), dtype=int)
    # Fill some random spots to make it realistic
    board[5, 5] = 1
    board[6, 6] = -1
    board[0, 0] = 1
    board[10, 10] = -1
    
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = HexPlanes.get_all_feature_planes(board, 1)
    end_time = time.perf_counter()
    
    return (end_time - start_time) / num_runs

def main():
    print("Preparing Latency Test...")
    
    # Load Model
    model = HexNet()
    model.eval()
    
    # Dummy Input
    # Batch size 1, 4 channels, 11x11 board
    dummy_input = torch.randn(1, 4, 11, 11)
    
    # 1. Feature Generation Latency
    print("\nTesting Feature Generation Latency (CPU)...")
    feat_time = test_feature_generation_latency()
    print(f"Average Feature Generation Time: {feat_time*1000:.4f} ms")
    
    # 2. CPU Inference Latency
    print("\nTesting Model Inference Latency (CPU)...")
    cpu_time = measure_latency("cpu", model, dummy_input)
    print(f"Average CPU Inference Time: {cpu_time*1000:.4f} ms")
    
    # 3. GPU Inference Latency
    if torch.cuda.is_available():
        print("\nTesting Model Inference Latency (GPU)...")
        gpu_time = measure_latency("cuda", model, dummy_input)
        print(f"Average GPU Inference Time: {gpu_time*1000:.4f} ms")
        
        print(f"\nSpeedup (CPU/GPU): {cpu_time/gpu_time:.2f}x")
    else:
        print("\nGPU not available, skipping GPU test.")

    # 4. JIT Compiled CPU Latency
    print("\nTesting JIT Compiled Model Inference Latency (CPU)...")
    try:
        jit_model = torch.jit.script(model)
        jit_cpu_time = measure_latency("cpu", jit_model, dummy_input)
        print(f"Average JIT CPU Inference Time: {jit_cpu_time*1000:.4f} ms")
    except Exception as e:
        print(f"JIT Compilation failed: {e}")

    # 5. JIT Compiled GPU Latency
    if torch.cuda.is_available():
        print("\nTesting JIT Compiled Model Inference Latency (GPU)...")
        try:
            jit_model_gpu = torch.jit.script(model).cuda()
            jit_gpu_time = measure_latency("cuda", jit_model_gpu, dummy_input)
            print(f"Average JIT GPU Inference Time: {jit_gpu_time*1000:.4f} ms")
        except Exception as e:
            print(f"JIT GPU test failed: {e}")

if __name__ == "__main__":
    main()