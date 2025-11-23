#!/usr/bin/env python
import torch

def check_gpu_support():
    if torch.cuda.is_available():
        print(f"\nGPU device(s) found!")
        print(f"  #GPUs: {torch.cuda.device_count()}")
        print(f"  Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("\nNO GPU device detected.")
check_gpu_support()
