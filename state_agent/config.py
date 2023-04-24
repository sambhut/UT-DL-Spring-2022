import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(8, device=device)
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

