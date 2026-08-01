import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
data_dir = Path('/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/processed')
run_dir = data_dir / 'TSeries-03042024-run02-054'

# Load data
F = np.load(run_dir / 'F.npy')
stat = np.load(run_dir / 'stat.npy', allow_pickle=True)
ops = np.load(run_dir / 'ops.npy', allow_pickle=True).item()

# 1. Plot raw fluorescence traces for a few cells
plt.figure(figsize=(10, 4))
for i in range(3):
    plt.plot(F[i, :1000], label=f'Cell {i}')
plt.title("Raw Fluorescence Traces (First 1000 frames)")
plt.xlabel("Frame")
plt.ylabel("Fluorescence")
plt.legend()
plt.savefig('raw_traces.png')
print("Saved raw_traces.png")

# 2. Plot mean image with cell locations
plt.figure(figsize=(8, 8))
plt.imshow(ops['meanImg'], cmap='gray')
for s in stat:
    plt.plot(s['med'][1], s['med'][0], 'c.', markersize=2)
plt.title("Mean Image with Detected Cells")
plt.savefig('mean_image_with_rois.png')
print("Saved mean_image_with_rois.png")
