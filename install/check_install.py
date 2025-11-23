import os
import sys
import subprocess

def cmd_version(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5).decode().strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

print(f"\nCheck installation")
print(f"  python           :: {sys.version}")
try:
    import numpy as np
    print(f"  numpy            :: {np.__version__}")
    import scipy as sp
    print(f"  scipy            :: {sp.__version__}")
    import sklearn as sk
    print(f"  scikit-learn     :: {sk.__version__}")
    import matplotlib
    print(f"  matplotlib       :: {matplotlib.__version__}")
    import torch
    print(f"  torch            :: {torch.__version__}")
    import torchaudio
    print(f"  torchaudio       :: {torchaudio.__version__}")
    import torchvision as torchvis
    print(f"  torchvision      :: {torchvis.__version__}")
    import tensorboard as tb
    print(f"  tensorboard      :: {tb.__version__}")

    os.environ["KERAS_BACKEND"] = "torch"   
    import keras
    print(f"  keras            :: {keras.__version__}")

    # Jupyter
    print(f"  jupyter lab      :: {cmd_version(['jupyter', 'lab', '--version'])}")
    print(f"  jupyter notebook :: {cmd_version(['jupyter', 'notebook', '--version'])}")


except Exception as err:
    print(f"\n  Invoking exception ... \n    {err}")
    sys.exit(f"    Goodbye!!")

print(f"Installation looks OK!")    


# Check whether GPU support for torch:
def check_gpu_support():
    if torch.cuda.is_available():
        print(f"\nGPU is available!")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("\nNO GPU support detected.")
check_gpu_support()

print(f"\nReady to go!")

