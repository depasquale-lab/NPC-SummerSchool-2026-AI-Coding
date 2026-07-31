# Day 1: Calcium Imaging Analysis with Agentic AI
## Two-Photon Neural Recording in Real Time

Learn to analyze real neural recordings from awake mice using Python and Claude AI. This 2-hour Friday evening exercise teaches you two core concepts: **ROI detection** (finding neurons) and **neuropil correction** (removing contamination).

---

## What You'll Do

### 🧠 **Block 1: Simple ROI Detection** (~30 min)
Learn how neurons are detected in raw imaging data.

**Your task**: 
- Write a simple ROI detection algorithm (Gaussian smoothing + threshold + connected components)
- Compare your results to Suite2p/Cellpose (professional deep learning pipeline)
- Understand *why* simple methods fail and deep learning is necessary

**Key finding**: Your simple algorithm finds 99 cells, but only 24 are real. Why? False positives in bright neuropil (2.4× brighter than real cells).

**Expected output**:
- Detection sensitivity: ~87.6%
- Precision: ~23.2%
- Why professionals use deep learning ✓

---

### 🔧 **Block 2: Neuropil Correction Factor** (~30 min)
Learn to remove contamination from neural signals.

**Your task**:
- Load corrected fluorescence: `F_corrected = F - α × Fneu`
- Test different correction factors (α = 0.2 to 1.2)
- Find the optimal factor for *your* dataset using three quality metrics:
  - F vs Fneu correlation
  - Signal variance
  - Fano factor (spike-to-spike variability)

**Key finding**: Standard literature value is α = 0.7, but it's not optimal for all datasets. For this dataset, optimal is α ≈ 0.547 (21.9% lower than standard).

**Expected results**:
- Correlation reduction: ~89%
- Quality score improvement: ~15%
- Conclusion: Don't blindly copy constants. Validate on *your* data. ✓

---

## Mathematical Underpinnings: Neuropil Correction

### The Problem
When imaging neurons, you measure fluorescence from:
1. **The target cell (F)** — what you want
2. **Surrounding neuropil (Fneu)** — contamination

The signal is: **F_measured = F + α_true × Fneu + noise**

### The Solution
Correct by subtracting the neuropil:

```
F_corrected = F - α × Fneu
```

**Why this works**: The neuropil (Fneu) is optically isolated during Suite2p processing, so it's a pure estimate of contamination. Multiplying by α (typically 0.7) removes the average contamination without over-correcting.

### Why α ≠ 0.7 for All Datasets
The contamination fraction varies based on:
- **Cellular packing density** — dense tissue = more neuropil signal
- **Dye distribution** — uneven loading = different α
- **Optical properties** — scattering depth affects contamination

**Solution**: Learn α from the data itself by minimizing:

```
α* = argmin(corr(F - α×Fneu, Fneu))
```

We want F and Fneu uncorrelated; when they are, contamination is removed.

---

## Optional: Spike Deconvolution (Friday 8-9pm)

If you complete Blocks 1–2, you can optionally explore **spike deconvolution** using NNLS (Non-Negative Least Squares):

### The Problem
Calcium rises and falls over **seconds**, but spikes last **milliseconds**. How do you recover spike times from slow fluorescence?

### The NNLS Solution

Assume fluorescence is a convolution of spikes with the calcium response kernel:

```
F(t) = ∑ spikes(τ) × kernel(t - τ) + baseline + noise
```

To recover spikes, solve the inverse problem:

```
min ||F - convolved_spikes||²₂ + λ × ||spikes||₁

subject to: spikes ≥ 0
```

**Parameters**:
- **λ** (regularization) — prevents overfitting. High λ = fewer spikes, less noise; low λ = more spikes, more noise
- **threshold** — final binarization: `spikes[spikes < threshold] = 0`

**Why NNLS?**
- Non-negative constraint makes biological sense (spikes can't be negative)
- L1 regularization promotes sparsity (natural spike structure)
- Simple alternative to Suite2p's OASIS algorithm

### NNLS vs OASIS
| Metric | NNLS | OASIS |
|--------|------|-------|
| Speed | Slow (~seconds/cell) | Fast (~ms/cell) |
| False positives | ~34% | ~15% |
| False negatives | ~0% | ~5% |
| When to use | Teaching / validation | Production |

---

## Setup Instructions

### Step 1: Log In to SCC

Open your terminal on your personal machine and SSH to the SCC:

```bash
ssh username@scc.bu.edu
```

Replace `username` with your SCC username (same as your BU username).

### Step 2: Navigate to Your Project Directory

```bash
cd /projectnb/npcr25/students/username
```

Replace `username` with your SCC username.

### Step 3: Clone This Repo

```bash
git clone https://github.com/anthropics/ai-coding-neuroscience.git
cd ai-coding-neuroscience
```

(Replace with actual repo URL when available)

### Step 4: Check Your Python Version

```bash
python3 --version
```

You should have **Python 3.13.8**. If not, load the module:

```bash
module load python/3.13.8
```

### Step 5: Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### Step 6: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- **numpy** — numerical computing
- **scipy** — signal processing & statistics
- **h5py** — load Suite2p .mat files
- **matplotlib** — plotting
- **jupyter** — notebooks
- **pytest** — testing

### Step 7: Start Jupyter (CPU-Only Mode)

On the SCC, **request a CPU-only job** (not GPU):

```bash
# Request an interactive CPU job for 4 hours
srun --partition=compute --nodes=1 --cpus-per-task=4 --mem=8G --time=04:00:00 --pty bash

# Start Jupyter
source .venv/bin/activate
jupyter notebook --ip=0.0.0.0 --no-browser
```

Then open the URL it prints (looks like `http://localhost:8888/?token=...`).

### Step 8: View Notebooks in Chrome or Safari

- **Chrome** or **Safari** are recommended (better Jupyter performance than Firefox)
- Open the Jupyter URL from Step 7
- Click `tutorials/day_1_core/simple_roi_detection.ipynb`

---

## Your Work on SCC

### Directory Structure

```
/projectnb/npcr25/students/username/
├── ai-coding-neuroscience/              ← Clone of this repo
│   ├── tutorials/day_1_core/            ← Notebooks you'll edit
│   ├── requirements.txt
│   └── README.md                        ← This file
│
└── DATA FILES (local, not in repo)      ← Use /projectnb/npcr25/shared/
    ├── run02-054/
    ├── run06-058/
    └── run03-055/
```

### Branching Workflow

Before you start editing:

```bash
cd ai-coding-neuroscience

# Create a branch for your work
git branch my-analysis-day1
git checkout my-analysis-day1

# OR in one command:
git checkout -b my-analysis-day1

# When you're done, push to GitHub:
git add tutorials/day_1_core/
git commit -m "Day 1: ROI detection + neuropil correction"
git push origin my-analysis-day1

# Then create a Pull Request on GitHub for peer review
```

---

## Data Overview

### Dimensions & Formats

The raw two-photon imaging data comes from **awake Thy1-jRGECO1a mice** (layer 2 cortex, spontaneous activity).

#### Per-Run Data
| Item | Format | Size | Typical Values |
|------|--------|------|-----------------|
| **F** (cell fluorescence) | .npy | 125 cells × 4535 frames | ~20-500 counts/frame |
| **Fneu** (neuropil) | .npy | 125 cells × 4535 frames | ~50-200 counts/frame |
| **iscell** (quality) | .npy | 125 cells | [0.0 to 1.0] |
| **spks** (spikes) | .npy | 125 cells × 4535 frames | [0 to 100+] |
| **stat** (ROI info) | .npy | 125 cells | cell location, size, depth |

#### Recording Parameters
| Parameter | Value |
|-----------|-------|
| **Frame rate** | 15 Hz |
| **Duration** | 5 min (run02-054) to 10 min (run06-058) |
| **Field of view** | 1024 × 1024 pixels |
| **Pixel size** | ~0.37 μm/pixel |
| **Total cells** | 125 (run02-054) to 168 (run03-055) |
| **Good cells** | 89% (quality ≥ 0.15 threshold) |

### File Locations

**Shared data** (accessible to all students):
```
/projectnb/npcr25/shared/imaging_data/
├── run02-054/
│   ├── F.npy
│   ├── Fneu.npy
│   ├── iscell.npy
│   ├── spks.npy
│   ├── stat.npy
│   └── ops.npy
├── run06-058/
├── run03-055/
└── ...
```

**Master data** (Martin's lab, 9 runs):
```
/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/processed/
├── TSeries-03042024-run01-053/
├── TSeries-03042024-run02-054/  ← Used in this exercise
├── TSeries-03042024-run03-055/
└── ... (6 more runs)
```

### Example: Loading Data in Python

```python
import numpy as np
from pathlib import Path

# Your SCC home directory
data_dir = Path('/projectnb/npcr25/shared/imaging_data/run02-054')

# Load Suite2p outputs
F = np.load(data_dir / 'F.npy')          # (125, 4535)
Fneu = np.load(data_dir / 'Fneu.npy')    # (125, 4535)
iscell = np.load(data_dir / 'iscell.npy') # (125,)
stat = np.load(data_dir / 'stat.npy', allow_pickle=True)  # (125,)

# Filter by quality (iscell ≥ 0.15)
good_cells = iscell[:, 0] >= 0.15
F_good = F[good_cells, :]
Fneu_good = Fneu[good_cells, :]

print(f"Loaded {F.shape[0]} cells, {F.shape[1]} frames")
print(f"Good cells: {F_good.shape[0]} ({100*F_good.shape[0]/F.shape[0]:.1f}%)")
```

---

## Milestones & Benchmarks

### Block 1 Milestones
- [ ] Write ROI detection code (Gaussian + threshold + connected components)
- [ ] Detect cells in the median image
- [ ] Compare to Suite2p detection (`stat.npy`)
- [ ] Measure sensitivity, precision, F1 score
- [ ] Plot overlays showing false positives

**Benchmark**: ~87% sensitivity, ~23% precision (simple method is limited—this is expected!)

---

### Block 2 Milestones
- [ ] Load fluorescence and neuropil from .npy files
- [ ] Implement correction: `F_corrected = F - alpha * Fneu`
- [ ] Test alpha values from 0.2 to 1.2
- [ ] Compute before/after correlations (z-scored)
- [ ] Find optimal alpha (minimizes F vs Fneu correlation)
- [ ] Plot corrected vs raw traces for example cells

**Benchmark**: 
- Correlation reduction: ~89% (±5%)
- Optimal alpha: ~0.55 (not 0.7!)
- Visual inspection: Raw shows Fneu contamination, corrected is clean

---

## Expected Results (From Our Analysis)

### ROI Detection Example
![ROI Detection Comparison](docs/figures/examples/real_stage1_roi_detection.png)

*Your simple algorithm (yellow circles) vs Suite2p (red circles). Note the false positives in bright neuropil.*

### Before/After Neuropil Correction
![Neuropil Correction](docs/figures/examples/real_stage3_correction.png)

*Top: raw fluorescence (pink/red). Bottom: corrected (blue/white). Note how contamination is removed.*

### Correlation Improvement
![Correlation Reduction](docs/figures/reference/correlation_before_after.png)

*Left: raw F vs Fneu are highly correlated (contaminated). Right: corrected F vs Fneu are uncorrelated (clean).*

### Live Imaging Preview
See the raw two-photon imaging with overlaid ROIs and activity:

**Video**: `instructor_materials/imaging_preview/raw_with_rois_activity_FIXED.mp4`

This shows:
- **Black/gray background** — raw intensity (median-subtracted)
- **Cyan circles** — detected ROIs from Suite2p
- **Red/orange heatmap** — fluorescence activity (slowly fading in)

Watch how:
1. ROIs are positioned on bright cell bodies
2. Neuropil (surrounding area) is dimmer
3. Activity pulses correspond to spikes

---

## Python Environment

### What's Pre-Installed

Your `.venv/` includes everything you need:

```
numpy==2.5.1           # Numerical arrays
scipy==1.18.0          # Signal processing
matplotlib==3.11.1     # Plotting
h5py==3.16.0           # Load .mat files
pytest==9.1.1          # Unit tests
jupyter==1.0.0         # Notebooks
```

### Using Claude AI in Your Code

When stuck, use Claude via:

1. **Claude Code IDE** (VSCode extension)
   - Install: VSCode → Extensions → "Claude Code"
   - Ctrl+K (or Cmd+K on Mac) to open Claude inline
   - Paste code, ask questions, get suggestions

2. **Claude Console** (web interface)
   - Go to console.anthropic.com
   - Create a new thread for your exercise
   - Share notebook code and ask for help

3. **API Key Usage** (if configured)
   - Set: `export ANTHROPIC_API_KEY=your-key`
   - Use in scripts to call Claude programmatically

---

## Troubleshooting

### "File not found: F.npy"
Check the path:
```bash
ls /projectnb/npcr25/shared/imaging_data/run02-054/F.npy
```
If it doesn't exist, the data hasn't been copied to your location yet. Ask an instructor.

### "ImportError: No module named h5py"
Activate the virtual environment:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### "My correlations didn't decrease"
Common mistakes:
1. **Wrong formula**: `F - alpha*Fneu` (not `F + alpha*Fneu`)
2. **Forgot to z-score**: Correlation needs normalized signals
3. **Used raw F twice**: Make sure you're comparing raw vs corrected, not raw vs raw
4. **Wrong indexing**: Data might be transposed—check shape

### "Shapes don't match"
The .npy files are typically (ncells, nframes). If you see (nframes, ncells), transpose:
```python
if F.shape[0] > F.shape[1]:
    F = F.T
```

### Jupyter Connection Issues
If Jupyter won't connect:
```bash
# Kill any existing Jupyter processes
pkill -f jupyter

# Start fresh
source .venv/bin/activate
jupyter notebook --ip=0.0.0.0 --no-browser
```

---

## Helpful Resources

### Reference Notebooks (Run These First)
1. **`tutorials/day_1_core/simple_roi_detection.ipynb`** — Step-by-step ROI detection walkthrough
2. **`tutorials/day_1_core/neuropil_correction_factor.ipynb`** — Interactive alpha optimization

### Testing Your Code
```bash
# Run all tests
pytest

# Run one test file with verbose output
pytest test_my_code.py -v

# Run one specific test
pytest test_my_code.py::test_correction_formula -v
```

### Useful Commands
```bash
# Monitor your job
squeue -u $USER

# Stop a Jupyter session gracefully
pkill -f jupyter

# Check available Python versions
module avail python

# List installed packages
pip list

# Deactivate your virtual environment
deactivate
```

---

## Git Workflow Summary

### First Time Setup
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@bu.edu"
```

### Create Your Branch
```bash
git checkout -b my-day1-analysis
```

### Make Changes, Commit, Push
```bash
# Edit tutorials/day_1_core/simple_roi_detection.ipynb
# ... do your analysis ...

git add tutorials/day_1_core/
git commit -m "Day 1: ROI detection + neuropil correction analysis"
git push origin my-day1-analysis
```

### Create Pull Request
On GitHub, click "Compare & pull request" and describe your analysis. Include:
- Results summary (sensitivity, precision, optimal alpha)
- Key findings (why simple methods fail, etc.)
- Screenshots of plots

---

## Questions?

### Recommended Order:
1. **Read this README** (you're here!)
2. **Run `tutorials/day_1_core/simple_roi_detection.ipynb`** (see the problem)
3. **Run `tutorials/day_1_core/neuropil_correction_factor.ipynb`** (see the solution)
4. **Write your own code** in a notebook or script
5. **Test your code** (pytest)
6. **Commit and push** to your branch

### If Stuck:
- Check data path: `ls /projectnb/npcr25/shared/imaging_data/`
- Review the reference notebook output
- Ask Claude: "My correlations didn't decrease. What's wrong?"
- Reach out to instructors on Slack

---

## Getting Help with Claude

You have **$500 of Claude API credit** for this exercise. Use it wisely:

**Good questions for Claude:**
- "I got shape mismatch (4535, 125) instead of (125, 4535). How do I transpose?"
- "My correlation didn't decrease after correction. Here's my code: [paste]. What's wrong?"
- "Explain neuropil correction in simple terms"
- "How do I write a unit test for my correction function?"

**Not ideal for Claude:**
- "Do my homework" (Claude will help explain, but won't just write code)
- Extremely long code dumps (paste 10-20 lines max)
- Questions answerable by reading the notebooks (read first, then ask Claude if confused)

---

## Citation

If you use this exercise or dataset, please acknowledge:

**Data**: Martin's lab, awake Thy1-jRGECO1a mice, layer 2 cortex, spontaneous activity, 15 Hz two-photon imaging

**Exercise**: Agentic AI Coding for Neuroscience, Anthropic Summer School, 2026

---

**Happy analyzing! 🧠📊**
