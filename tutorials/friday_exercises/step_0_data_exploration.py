import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
data_dir = Path('/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/processed')
run_dir = data_dir / 'TSeries-03042024-run02-054'

# Load data
F = np.load(run_dir / 'F.npy')
Fneu = np.load(run_dir / 'Fneu.npy')
iscell = np.load(run_dir / 'iscell.npy')
stat = np.load(run_dir / 'stat.npy', allow_pickle=True)
ops = np.load(run_dir / 'ops.npy', allow_pickle=True).item()

# Filter good cells
good_cells = iscell[:, 0] == 1
F_good = F[good_cells, :]

print(f"Total cells: {F.shape[0]}")
print(f"Good cells: {F_good.shape[0]}")
print(f"Frames: {F.shape[1]}")
