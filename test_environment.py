import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(f"MPS device found: {x}")
    print(f"PyTorch built with MPS support: {torch.backends.mps.is_built()}")
else:
    print("MPS device not found.")
    if not torch.backends.mps.is_built():
        print("PyTorch was NOT built with MPS enabled.")
    else:
        print("MPS is built but not available (e.g., macOS version too old or no M-series chip).")