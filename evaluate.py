import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from models.generator import Generator
from config import *

def generate(model_path, num_samples=64, device="cpu"):
    net_g = Generator().to(device)
    net_g.load_state_dict(torch.load(model_path, map_location=device))
    net_g.eval()

    noise = torch.randn(num_samples, Z_DIM, 1, 1, device=device)
    with torch.no_grad():
        fakes = net_g(noise)

    grid = vutils.make_grid(fakes, normalize=True, nrow=8)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.title("Fake samples")
    plt.show()

if __name__ == '__main__':
    generate('output/net_g_10.pth', device='cuda' if CUDA and torch.cuda.is_available() else 'cpu')