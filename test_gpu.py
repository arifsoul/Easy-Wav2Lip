# Easy-Wav2Lip/test_gpu.py

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
    print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("CUDA version:", torch.version.cuda)
else:
    print("Tidak ada GPU yang terdeteksi.")
