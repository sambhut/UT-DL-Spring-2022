import torch

device = None
if torch.backends.mps.is_available():
    print("MPS device  found.")
    device = torch.device("mps")
    x = torch.ones(8, device=device)
    print(x)
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"