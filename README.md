# Friday Exercises: Calcium Imaging Data Processing with Agentic AI
## From Raw Imaging to Neural Spike Trains

**Roadmap**: This document has two parts. First, **setup** (getting onto the SCC, installing an AI coding assistant, cloning the repo) — skip ahead if you've already done this. Second, **the three exercises** themselves, starting at [The Exercises](#the-exercises-warmup--main--challenge). If you just want to know what you'll actually be doing, jump there now and come back for setup afterward.

💬 **First thing to do once Cline is set up**: ask it to read this README (`README.md`) in full. That gives it the full context for everything you'll ask it afterward, instead of guessing from a single pasted snippet.

### What This Exercise Is About

You'll learn to analyze **real two-photon calcium imaging recordings** from awake mice using Python and AI-assisted coding (Cline + Gemini). The data comes from cortical neurons expressing genetically-encoded calcium indicators (jRGECO1a), imaged at 15 Hz during spontaneous activity.

**The challenge**: Raw imaging data looks like noisy grayscale video. Your job is to extract **when each neuron fired spikes** — the fundamental unit of neural communication. Here's what you're analyzing:

<img src="assets/imaging_preview.gif" alt="Two-Photon Imaging with ROIs and Activity" width="800">

*Raw imaging frames (gray) with cyan ROI circles (Suite2p detected neurons) and red/orange activity heatmap (fluorescence).*

**What you're seeing**: The grayscale background is raw pixel intensity from the actual recording. Overlaid are cyan circles marking detected cell bodies (~10–15 μm diameter in this dataset), and a red/orange heatmap showing each cell's own fluorescence rising and falling over time. Watch individual cells light up and dim at different moments — that's the calcium signal, and it's slow and blurred compared to the instantaneous spikes that caused it.

Two things to notice, each pointing at one of this exercise's problems:
- **Not every cyan circle is a "clean" detection.** Some bright regions are neuropil contamination, not the neuron itself — Exercise 1 removes this contamination, and Exercise 3 explores why finding the cells in the first place is hard.
- **The slow rise and fall you're watching is not the spike train.** It's the *convolved* response — a blurred version of fast, discrete spikes. Exercise 2 is about inverting that blur to recover the original spike times.

This exercise walks through three interconnected problems, in this order:

1. **Neuropil Correction** (Exercise 1) — How do we remove contamination from nearby tissue?
2. **Spike Deconvolution** (Exercise 2) — When did spikes occur, given the slow calcium dynamics we just saw?
3. **ROI Detection** (Exercise 3) — Which pixels belong to which neurons, in the first place?

### Why This Matters

**For neuroscience**: Neurons communicate through spike timing — the precise moments when they fire. Extracting accurate spike times is foundational because it reveals:
- **What neurons encode**: A visual cortex neuron might fire in response to oriented edges. A motor cortex neuron might fire before a specific movement. Without spike timing, you can't answer "what does this neuron do?"
- **How circuits work**: Population activity patterns show whether neurons coordinate (synchronized firing) or compete (mutual inhibition). This reveals circuit function at the network level.
- **How learning happens**: The timing of spikes matters for synaptic plasticity. Spike-timing-dependent plasticity (STDP) is a fundamental learning rule — spikes separated by milliseconds have opposite effects on synapse strength.
- **Disease signatures**: Abnormal spike patterns appear in epilepsy, autism, and neurodegeneration before visible behavioral changes.

In short: **you can't do modern neuroscience without accurate spike times.**

**For you (and your career)**: This exercise teaches the *exact workflow* used in labs worldwide:
- You'll encounter real biological problems (noise, contamination, slow sensor dynamics) that don't appear in textbooks
- You'll design algorithms by understanding tradeoffs (fast but less accurate? slow but precise? what's the right balance?)
- You'll validate your work rigorously — first on synthetic data where you know the ground truth, then on real neurons
- You'll compare yourself against the standard (Suite2p's spike inference) so you understand where your method is competitive and where it falls short
- You'll learn to read and understand scientific code — an essential skill in research

**For AI + science**: You'll see how agentic AI *augments* (not replaces) human understanding:
- When your optimization doesn't converge, ask Cline "Why isn't this working?" It'll suggest debugging steps, explore numerical issues, and help you iterate quickly
- When you encounter new concepts (what's a Toeplitz matrix? why is it useful?), ask Cline for intuitive explanations
- You'll write the *logic* and *math*, but offload boilerplate (building matrices, plotting, I/O) so you stay focused on what matters
- You'll validate implementations: Cline can help test whether your code matches the paper's equations
- **Result**: faster development, deeper understanding, less busywork

### What You'll Build

You'll write **three data-processing pipelines**, each solving one piece of the puzzle:

**Exercise 1: Neuropil Removal** (Warmup)
- **Input**: Raw fluorescence traces (F) and neuropil contamination (Fneu)
- **Output**: Cleaned fluorescence and understanding of how correction works
- **What you'll learn**: How preprocessing works, and how to measure whether it actually worked (correlation between corrected F and Fneu, before vs. after)
- **Real-world parallel**: Every lab applies neuropil correction. This is a standard first step in calcium imaging pipelines.

**Exercise 2: Spike Deconvolution** (Main)
- **Input**: Corrected fluorescence from Exercise 1
- **Output**: Spike times recovered from fluorescence, using a calcium kernel timescale you estimate for each cell
- **What you'll learn**:
  - First, on data you generate yourself: create a fluorescence trace from a spike train with a known timing, then recover it — so you can check whether your method actually works before trusting it on real neurons
  - Then, on real neurons: estimate each cell's own calcium decay time, recover its spikes, and compare against Suite2p's spike inference
- **Key insight**: Suite2p was run with a single fixed decay timescale for every cell in this dataset. Real neurons vary — this exercise estimates that timescale directly from each cell's own data instead.
- **Real-world parallel**: This is the core of calcium-imaging analysis. Getting spikes right determines everything downstream.

**Exercise 3: ROI Detection** (Challenge)
- **Input**: The real 2D field-of-view image (not a per-cell trace — an actual picture of the tissue)
- **Output**: Detected cell locations from a simple threshold-based detector
- **What you'll learn**: Why a simple, single-image threshold approach struggles to distinguish real cells from bright neuropil, and why Suite2p's approach (which exploits activity *dynamics* over the whole movie, not just one static image) does much better
- **Real-world parallel**: Finding neurons is the first step, but it's surprisingly hard. Understanding why teaches you to respect the complexity of biological data.

### By the End

You'll have **three working data-processing pipelines**, each self-contained:
- Code that takes raw imaging → cleaned spikes (the full pipeline)
- Detailed understanding of *why* each step exists (not just "we do it because papers do")
- Quantitative benchmarks showing how your methods compare to the standard (Suite2p)
- Intuition about the tradeoffs in algorithm design (speed vs. accuracy, sensitivity vs. specificity)
- Hands-on experience with numerical optimization, inverse problems, and data validation — skills that transfer to any data science problem

---

## Quick Start: Get Set Up on SCC

### Step 1: Access the SCC and Launch VS Code Server

⚠️ **Browser requirement**: Use **Chrome** or **Safari** for best compatibility with OnDemand and VS Code Server.

Go to: [https://scc-ondemand2.bu.edu/](https://scc-ondemand2.bu.edu/)

Log in with your BU username/password. This opens a web-based interface — no terminal needed.

Once logged in to OnDemand:

1. Click **Interactive Apps** (left sidebar)
2. Click **VS Code Server**
3. Fill in the form with these settings:

   | Setting | Value |
   |---------|-------|
   | **Number of hours** | 6 |
   | **Number of cores** | 1 |
   | **Number of GPUs** | 0 |
   | **Project** | npcr25 |
   | **Additional modules to load** | python3/3.13.8 |
   | **Working Directory** | `/projectnb/npcr25/students/username` |

   ⚠️ **Important**: Replace `username` with your SCC username (same as your BU login)

4. Click **Launch**
5. Wait for the session to start
6. Click the **VS Code Server** button that appears
7. VS Code opens in your browser — you're now on the SCC with everything pre-loaded!

---

### Step 2: Set Up an AI Coding Assistant

**Cline + Gemini** is the default path below, and what the rest of this guide assumes you're using. Already comfortable with another AI tool — Claude, ChatGPT, etc.? You're welcome to use that instead; skip to Step 3.

### Installation Steps

1. Click on the **Extensions** toolbar item in VSCode and search for **"Cline"** and install.
2. Click on the **"robot"** icon that should appear in the left toolbar to open Cline.
3. Choose **"Bring my own API key"**.
4. For API provider choose **"OpenAI Compatible"**.
5. For Base URL type: `https://generativelanguage.googleapis.com/v1beta/openai/`
6. Navigate to [https://aistudio.google.com/api-keys](https://aistudio.google.com/api-keys) (make sure you are logged in as your personal Google account) and click **"Create API key"**. This will create a project and API key.
7. For OpenAI Compatible API key, copy-paste the key you just created.
8. For model type: `gemini-3.1-flash-lite`

### How to Use Cline

Once installed and configured:
- Open Cline in VSCode (robot icon in left sidebar)
- Paste your code or describe what you need help with
- Ask questions like: "My correlations didn't decrease. What's wrong?" or "Explain neuropil correction"
- Cline will help you debug, explain code, and suggest improvements

### Usage Limits

You get **500 requests per day** with `gemini-3.1-flash-lite`. Check usage at:
[https://aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit)

If you run out:
- Switch to `gemma-4-26b-a4b-it` (1.5K requests/day, less capable)
- Contact an instructor for unlimited access via our Google Cloud Project

---

### Step 3: Clone This Repository (via Cline)

Now that Cline is installed, use it to clone the repo.

**In the VS Code terminal**, ask Cline:

> "Clone the repository: https://github.com/depasquale-lab/NPC-SummerSchool-2026-AI-Coding.git"

Cline will execute these commands for you. Then verify it worked by running `ls README.md` in a terminal. You should see the README.md file confirming it cloned successfully.

**Next, create a working branch** for your exercise work. Ask Cline:

> "Create a new git branch for my work. Call it `username`" 

(Replace `username` with your SCC username.) This creates your personal branch where you'll commit your progress. Then confirm with `git branch` in a terminal. 

**Important**: After you complete progress on each exercise, commit and push your work:

```bash
git add -A
git commit -m "Exercise N complete: <brief description>"
git push origin username
```

Pushing after each exercise ensures your work is backed up and instructors can see your progress.

---

### Step 4: Set Up Python Environment (via Cline)

Now that Cline is installed, use it to set up your Python environment.

**In the VS Code terminal**, ask Cline:

> "Set up a Python virtual environment with all required packages. Run these commands:
> ```
> python3 -m venv .venv
> source .venv/bin/activate
> pip install --upgrade pip
> pip install -r requirements.txt
> ```
> Then verify it worked by running `python3 -c "import numpy; print('Success!')"`"

Cline will execute these commands for you. You should see `(.venv)` in your terminal prompt when done.

---

## Your Work Directory Structure

```
/projectnb/npcr25/students/username/
├── NPC-SummerSchool-2026-AI-Coding/     ← Cloned repo (contains only README.md + assets/)
│   ├── README.md                        ← This guide
│   └── assets/                          ← Images (GIFs, PNGs)
│
├── tutorials/day_1_core/                ← YOU CREATE with Cline
│   ├── step_0_data_exploration.ipynb
│   ├── exercise_1_neuropil_correction.ipynb
│   ├── exercise_2_spike_deconvolution.ipynb
│   └── exercise_3_roi_detection.ipynb
│
├── my_analysis.py, notes.txt, etc.      ← YOUR work files
│
└── (data accessed from /projectnb2/npcr25/projects/two_photon/.../processed/)
```

---

## Data Overview

### What You're Looking At

This is a real two-photon calcium imaging recording: a live, awake mouse, imaged through a cranial window, with neurons in layer 2 of cortex expressing a genetically-encoded calcium indicator (jRGECO1a) that gets brighter when a neuron's intracellular calcium rises — which happens when it fires. The raw output of an experiment like this is a video: thousands of grayscale frames, one per timepoint, at 15 Hz.

That raw video has already been run through **Suite2p** (see below) to produce the processed files you'll actually load: cell locations, per-cell fluorescence traces, and a reference set of inferred spike times. You're working from Suite2p's *output*, not the raw video — except in Exercise 3, which also uses the field-of-view image (`ops['meanImg']`). The raw movie exists on disk too, if you're curious (see paths below), but you won't need it for any exercise.

**Use this dataset — everyone should.** `TSeries-03042024-run02-054` is what generated every number, benchmark, and figure in this README. If you use a different dataset, your results won't match what's described here (fine if you're exploring on your own, but the "Expected result" sections won't apply).

**Where the data lives**:
- **Processed** (what you'll actually load): `/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/processed/TSeries-03042024-run02-054/`
- **Suite2p's spike inference** (your reference in Exercise 2): `/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/Suite2P-inferred-spikes/TSeries-03042024-run02-054/`
- **Raw acquisition** (not needed for the exercises): `/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/2photon/TSeries-03042024-run02-054/` — the microscope's original output, as multi-gigabyte OME-TIFF files plus acquisition metadata

**Further reading**:
- [Suite2p on GitHub](https://github.com/MouseLand/suite2p) — the pipeline that processed this data
- [Suite2p ROI detection docs](https://suite2p.readthedocs.io/en/latest/roidetection/)
- [Suite2p ROI extraction docs](https://suite2p.readthedocs.io/en/latest/roiextraction/)
- [Suite2p deconvolution docs](https://suite2p.readthedocs.io/en/latest/deconvolution/)

### The Dataset: run02-054

**Recording specs**:
- **Source**: Awake Thy1-jRGECO1a mouse, layer 2 cortex, spontaneous activity
- **125 neurons** × 4535 frames (5.04 minutes @ 15 Hz)
- **90% good quality** (113 cells passing iscell ≥ 0.15 threshold)
- **Cell size**: detected cell bodies are ~10–15 μm in diameter (measured from this dataset's own ROIs); pixels are ~0.4 μm
- **Well-characterized**: diverse cell types and firing rates

### What is Suite2p?

[**Suite2p**](https://github.com/MouseLand/suite2p) is the industry-standard software for processing two-photon imaging data. It's maintained by the Stringer Lab and used in hundreds of neuroscience papers worldwide.

**What Suite2p does**:
1. **ROI detection** ([docs](https://suite2p.readthedocs.io/en/latest/roidetection/)): Suite2p offers three detection algorithms, selected by an `algorithm` setting: **Sparsery** (the default), Sourcery, and Cellpose. Sparsery is a matrix decomposition that looks for spatially-compact, temporally-sparse sources directly in the movie (i.e. it exploits *when* and *where* fluorescence changes over time, not just a single static image) — this is the method that produced the ROIs in this dataset. Cellpose, by contrast, is a deep-learning anatomical segmentation model that works on a static summary image, not the movie — it's an available option, but was **not** used here.
2. **Fluorescence extraction** ([docs](https://suite2p.readthedocs.io/en/latest/roiextraction/)): Measures raw fluorescence (F) from each detected ROI
3. **Neuropil measurement** ([docs](https://suite2p.readthedocs.io/en/latest/roiextraction/)): Measures fluorescence (Fneu) from the surrounding neuropil tissue
4. **Spike inference** ([docs](https://suite2p.readthedocs.io/en/latest/deconvolution/)): Applies its OASIS-based spike inference algorithm to recover spike times from fluorescence
5. **Quality scoring**: Assigns a quality-of-detection index to every cell, rating confidence that each ROI is a real neuron

The result is a folder full of `.npy` files containing all the processed data.

### The Files You'll Load

**All the data in this exercise has already been processed by Suite2p.** You're not reimplementing Suite2p — you're learning how its pieces work by implementing and validating them yourself. Here's what each file contains:

**`F.npy`** — **Raw fluorescence** from each detected ROI
- Shape: 125 cells × 4535 frames
- What it is: The sum of all photons detected in each ROI during each frame
- Range: ~170–5550 counts/frame (varies across cells and time)
- Why you need it: This is the actual measurement you'll work with. It's noisy, slow (calcium dynamics), and contaminated (neuropil signal mixed in)

**`Fneu.npy`** — **Neuropil fluorescence** (the contamination signal)
- Shape: Same as F (125 cells × 4535 frames)
- What it is: Fluorescence from the tissue surrounding each ROI, measured from a surround mask (padded around the ROI, excluding other detected cells)
- Range: ~350–3730 counts/frame (often as bright or brighter than the cell signal!)
- How Suite2p measures it ([docs](https://suite2p.readthedocs.io/en/latest/roiextraction/)): For each ROI, Suite2p builds a "neuropil mask" by: (1) padding the ROI outward by a fixed number of pixels to exclude the cell itself, (2) growing a rectangular (or circular, if configured) region until it contains enough non-cell pixels, (3) excluding pixels belonging to other detected ROIs, and (4) averaging fluorescence across these neuropil pixels
- Why you need it: This is the contamination you'll remove in Exercise 1. By measuring Fneu separately, Suite2p gives you a direct estimate of the neuropil signal contaminating F. Without correction, ROIs show much higher correlation with each other due to spatially-invariant neuropil surges affecting all cells simultaneously. The correction Suite2p applied to this dataset is **F' = F - 0.7 × Fneu**. Suite2p's own documentation shows 0.7 as the example `neucoeff` value, and this dataset's own saved processing settings confirm 0.7 is exactly what was used here.

**`iscell.npy`** — **Cell quality scores**
- Shape: 125 cells × 2 columns
- Contains: A quality-of-detection index assigned to every ROI by Suite2p's built-in classifier (column 0: quality score, column 1: predicted class label)
- Threshold: Use iscell[:, 0] >= 0.15 to filter for likely "real cells" and exclude artifacts
- Why you need it: Not all detected ROIs are real neurons. Some are neuropil artifacts or background noise. This quality score helps you distinguish real cells from false detections. At threshold 0.15, you keep ~90% of the 125 cells.

**`spks.npy`** — **Suite2p's spike inference** (the reference you'll compare against)
- Shape: 125 cells × 4535 frames
- What it is: Spike amplitudes recovered by Suite2p's OASIS-based spike inference algorithm, run with a single fixed calcium timescale (τ = 1.0s) for every cell
- Range: 0–1543. Note this is **not** a sparse 0/1 spike train — many frames have small nonzero values, so a raw ">0" threshold captures far more than just "obvious" spike events. Comparing against it requires peak-detection, not simple thresholding.
- Why you need it: This is your benchmark. After you implement deconvolution in Exercise 2, you'll compare your spike detections against this reference.
- **Location note**: These spikes are stored in a separate directory: `/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/Suite2P-inferred-spikes/TSeries-03042024-run02-054/spks.npy`.

**`stat.npy`** — **ROI metadata**
- Shape: 125 cells
- What it contains: For each ROI — pixel coordinates (`xpix`, `ypix`), median location (`med`), size (`npix`), and other shape descriptors
- Why you need it: To visualize where cells are in the image, filter by size/location, or (in Exercise 3) compare your own detections against Suite2p's ROI positions

**`ops.npy`** — **Processing parameters and the field-of-view image**
- Contains Suite2p's run configuration (frame rate, neuropil coefficient, kernel timescale, etc.) plus `ops['meanImg']` — the actual 2D mean fluorescence image of the field of view (1024×1024 pixels for this dataset)
- Why you need it: This is what Exercise 3 uses as the "picture" to detect cells in. **`F.mean(axis=1)` is not a substitute** — F is already one trace per detected cell, not spatial pixel data.

### How You'll Compare to Suite2p

In each exercise, you'll implement a piece of the Suite2p pipeline and validate it:

- **Exercise 1**: You'll apply Suite2p's own neuropil correction (α = 0.7) and measure how much it reduces contamination.
- **Exercise 2**: You'll implement L1-regularized deconvolution with per-cell calcium kernels. Suite2p uses a different algorithm (OASIS) with one fixed kernel for every cell. You'll compare your spikes against Suite2p's and measure agreement.
- **Exercise 3**: You'll write a simple single-image threshold detector. Suite2p's default detects cells using movie dynamics (Sparsery), not a single static image. You'll see why that distinction matters.

**The goal**: Not to beat Suite2p (it's already well-engineered), but to understand how it works and develop intuition for algorithm design. When you later use Suite2p for your own research, you'll know exactly what it's doing under the hood.

### Loading Data in Python

```python
import numpy as np
from pathlib import Path

# Shared data directory with processed outputs
data_dir = Path('/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/processed')
run_dir = data_dir / 'TSeries-03042024-run02-054'

# Load fluorescence and cell info from processed directory
F = np.load(run_dir / 'F.npy')              # (125, 4535) — raw fluorescence
Fneu = np.load(run_dir / 'Fneu.npy')        # (125, 4535) — neuropil fluorescence
iscell = np.load(run_dir / 'iscell.npy')    # (125, 2) — cell quality scores
stat = np.load(run_dir / 'stat.npy', allow_pickle=True)  # (125,) — ROI locations
ops = np.load(run_dir / 'ops.npy', allow_pickle=True).item()  # dict — run config + meanImg

# Load Suite2p's spike inference from a separate directory
spks_dir = Path('/projectnb2/npcr25/projects/two_photon/Ex1_jRGECO1a_ResonantScanning/Suite2P-inferred-spikes')
spks = np.load(spks_dir / 'TSeries-03042024-run02-054' / 'spks.npy')  # (125, 4535)

# Filter by quality (iscell ≥ 0.15)
good_cells = iscell[:, 0] >= 0.15
F_good = F[good_cells, :]
Fneu_good = Fneu[good_cells, :]

# Counts after quality filtering
print(f"Total cells: {F.shape[0]}")
print(f"Good cells: {F_good.shape[0]} ({100*F_good.shape[0]/F.shape[0]:.1f}%)")
print(f"Frames: {F.shape[1]} ({F.shape[1]/15/60:.1f} minutes at 15 Hz)")
```

---

# The Exercises: Warmup → Main → Challenge

⚠️ **These exercises are designed for students new to Python and data analysis**. They guide you through real research pipelines step-by-step.

**If you already know Python/data science**: Feel free to use your own approaches! Work with different data, implement advanced variants, parallelize code, or extend the exercises. Use any AI tool you prefer (Cline, Claude, ChatGPT, etc.) — the goal is learning the neuroscience and algorithms, not following a script.

---

This exercise has **three parts, designed to build from simple to complex** — plus a Step 0 to get oriented first:

0. **Step 0**: Load the data and look at it — no analysis yet
1. **Exercise 1 (Warmup)**: Neuropil Removal — apply correction, compute metrics, visualize
2. **Exercise 2 (Main)**: Spike Deconvolution — solve inverse problem, estimate kernels per-cell, compare to Suite2p
3. **Exercise 3 (Challenge)**: ROI Detection — understand why movie dynamics beat a static-image threshold

You can do all three, or focus on Exercise 2 for depth.

---

## Step 0: Look at the Data First

Before Exercise 1, just load the files and look at them — no analysis, nothing to get "right."

**Deliverable**: Load `F`, `Fneu`, and `ops['meanImg']`. Plot a few raw fluorescence traces over time, and plot the mean image with cell locations marked (from `stat`, the `med` field).

**What you should see**:

![Raw fluorescence traces](assets/step0_raw_traces.png)

*A few cells' raw fluorescence over 100 seconds. Notice the slow rises and falls — not sharp spikes. That slowness is the whole problem Exercise 2 solves.*

![Mean image with detected cells](assets/step0_mean_image_with_rois.png)

*The actual field of view, averaged over the whole recording, with cyan dots marking Suite2p's detected cells. A grainy grayscale image with round bright blobs, dark blood-vessel-like curves, and dots landing on most (not all) of the bright blobs.*

If your plots look roughly like this, you're oriented and ready for Exercise 1.

💬 Not sure how to load an `.npy` file, or what `ops['meanImg']` even is? Ask Cline — that's exactly what it's there for.

---

## Exercise 1: Neuropil Removal (Warmup)

### The Problem

Fluorescence measured from a cell body comes from **two sources**: the cell itself (signal) and the surrounding tissue — dendrites, glia, other cells (contamination). Light from that surrounding tissue bleeds into your measurement because the microscope's optics aren't perfectly sharp.

$$F_{\text{obs}} = F_{\text{cell}} + \alpha \times F_{\text{neuropil}} + \text{noise}$$

Without correcting for this, a burst of activity in the surrounding tissue can look like activity in your cell — even if the cell did nothing.

**How Suite2p estimates the contamination**: You can't separate "cell" from "neuropil" within the ROI's own pixels, so Suite2p measures the neuropil separately, from pixels just outside the ROI (excluding any other detected cells). This gives `Fneu` — an estimate of the same contaminating signal that's leaking into `F`. That's why subtracting a scaled copy of it (`F - α × Fneu`) removes the contamination.

### The Existing Solution

Suite2p measured this dataset's neuropil and applied this correction ([docs](https://suite2p.readthedocs.io/en/latest/roiextraction/)):

$$F_{\text{corrected}} = F_{\text{obs}} - 0.7 \times F_{\text{neuropil}}$$

α = 0.7 (Suite2p calls this `neucoeff`) is exactly what this dataset's own saved processing settings show was used — not a guess or an approximation. You'll apply that same correction and measure how well it works.

### Deliverable

Apply `F_corrected = F - 0.7 * Fneu` to the good-quality cells (`iscell ≥ 0.15`), then show it worked: report the correlation between F and Fneu before vs. after correction, and plot a few example cells.

⚠️ Compute the correlation **per cell, then average** — not by pooling every cell's data into one big correlation. Pooling mixes in brightness differences *between* cells (which the correction can't fix) and makes the contamination look worse than it is.

💬 If your correlation isn't dropping the way you'd expect, describe what you're seeing to Cline and ask it to help you debug — this is a common place to get stuck on an off-by-one or an unintended broadcast.

**Expected result** (this dataset):
- Mean per-cell correlation: **0.44 → -0.13** (a **71.6%** reduction)
- The correction overshoots slightly rather than landing exactly on zero — an honest result, not a claim that α = 0.7 is perfect for every dataset
- Corrected traces look visibly cleaner, with sharper individual events

### What You'll See

![Neuropil correction: raw vs. corrected traces](assets/exercise1_neuropil_correction.png)

*Three example cells (rows), each shown two ways on the same z-scored scale: raw F vs. Fneu (left) and corrected F vs. Fneu (right). In the left column, the background wobble in blue tracks the red dashed neuropil trace closely. In the right column, that shared wobble is largely gone — the sharp transients that remain in green are essentially independent of Fneu.*

![Per-cell correlation before vs. after correction](assets/exercise1_per_cell_correlation.png)

*Every one of the 113 good cells, not just the average. Left: each cell's own F-vs-Fneu correlation before (blue) and after (green) correction, sorted by the raw value. Right: the same data as a scatter — every point falls below the y=x line, meaning the correction reduces correlation for every single cell, not just on average.*

---

## Exercise 2: Spike Deconvolution (Main)

### The Problem

Calcium indicators respond **slowly** to spikes. A single spike (~1 ms) triggers a fluorescence rise over 10–100 ms that decays over 100–1000 ms. When neurons fire in bursts, those slow responses overlap and individual spikes become invisible in the raw trace.

$$F(t) = \text{baseline} + \sum_{\text{spikes}} h(t - t_s) + \text{noise}, \qquad h(t) = \exp(-t/\tau)$$

Given the observed $F(t)$, you must **invert** this to recover the spike times.

### The Existing Solution

Suite2p's algorithm, **OASIS** ([docs](https://suite2p.readthedocs.io/en/latest/deconvolution/)), assumes exponential decay and runs non-negative deconvolution in milliseconds per cell. Its documentation shows τ = 1.0s as an example calcium timescale, and this dataset's own saved processing settings confirm it was actually run with **one fixed τ = 1.0s for every cell** — a simplification, since real neurons don't all share identical calcium kinetics.

**Why not just reimplement OASIS?** OASIS is a specialized, carefully optimized algorithm from its own dedicated research codebase — reproducing it exactly is a substantial engineering project on its own, not what this exercise is asking of you. Instead, you'll solve the *same underlying inverse problem* — recover spikes from a blurred fluorescence trace — with a simpler, more general tool (L1-regularized least squares) that you can build from first principles. That's the point: understanding the problem OASIS solves and building a working (if less polished) solution yourself, not reproducing OASIS's internals.

**Your task**: solve the same inverse problem explicitly, using a kernel timescale you estimate separately for each cell.

### Deliverable 1 — Synthetic Validation

**Method**: Generate a spike train from a Poisson process (~1 Hz, with a refractory period so spikes don't land right on top of each other), convolve it with an exponential calcium kernel (τ = 1.0s) to make synthetic fluorescence, and add shot + Gaussian noise. Recover the spikes by solving the inverse problem, then score sensitivity/precision/F1 against the ground truth you made.

⚠️ Two things that will silently break this:
- **Use a non-negative Lasso** (`sklearn.linear_model.Lasso(positive=True)`) for the actual L1 sparsity penalty — not `scipy.optimize.nnls` plus a smoothness penalty. Spikes are sparse impulses, not smooth curves; a smoothness penalty fights against recovering them.
- **Match peaks, not frames.** The recovered trace spreads across several frames around each real event. Find peaks in it and match each to the nearest true spike within a small tolerance window — comparing frame-by-frame manufactures false positives out of one event's own shoulders.

💬 This is the most mathematically involved exercise — if the inverse problem, the Toeplitz matrix, or why the L1 penalty matters doesn't click, ask Cline to walk through it with a concrete small example (e.g. 3 spikes, a short kernel). Also a good exercise to ask Cline to help tune `alpha` if your solver returns all zeros or way too many spikes.

### Deliverable 2 — Real Data

**Data**: Use a subset of run02-054 — e.g. the first 30 good-quality cells and the first 2000 frames (~2.2 minutes). The deconvolution below solves an $n_{\text{frames}} \times n_{\text{frames}}$ system per cell, so runtime grows fast with both cell count and frame count; a subset keeps this tractable. The benchmarks below were measured on exactly this subset — running on the full 113 cells × 4535 frames will work but will be much slower, and your numbers may shift somewhat since more/different cells are included.

**Method**: For each cell in your subset, estimate its own τ from the autocovariance of its fluorescence trace (fit the exponential decay slope across several lags — don't assume Suite2p's fixed 1.0s applies to every cell). Deconvolve real fluorescence with that cell's own kernel using the same Lasso approach as Deliverable 1, then score agreement against Suite2p's spike inference (`spks.npy`) using the **same sensitivity/precision/F1 metrics as Deliverable 1**.

⚠️ Suite2p's spikes are *not* a sparse 0/1 train — thresholding at ">0" isn't meaningful. Find peaks in it the same way you find peaks in your own trace, and don't switch to a different metric like Jaccard, which hides whether you're over- or under-detecting relative to Suite2p.

**Expected result** (this dataset):
- **Deliverable 1 (synthetic, SNR ≈ 3)**: F1 ≈ 0.34 (sensitivity 32%, precision 37%) — modest. Deconvolving overlapping, noisy transients is a genuinely hard problem; don't expect near-perfect recovery from a basic solver.
- **Deliverable 2 (real data)**: per-cell τ ranges ~0.17–2.3s, median ≈ 0.41s — clearly not one-size-fits-all, and mostly below Suite2p's fixed 1.0s. Against Suite2p's spike inference: F1 ≈ 0.73 (sensitivity 88%, precision 68%) — better agreement than the synthetic case, likely because real transients are larger relative to noise, and per-cell kernels fit real heterogeneity better than one global kernel.

### Implementation Notes

- **The inverse problem**: minimize $\frac{1}{2n}\|F - Hs\|_2^2 + \lambda\|s\|_1$ subject to $s \geq 0$, where $H$ is the Toeplitz convolution matrix built from the kernel
- **Use L1 sparsity, not smoothness.** Most timepoints have no spike — a penalty on $s$ itself matches that. A penalty on *differences between neighbors* pushes toward smooth, spread-out solutions, the opposite of what a spike train looks like.
- **Match peaks, not frames.** The kernel spans many frames, so one real event produces a broad bump — comparing that bump frame-by-frame against a single ground-truth frame manufactures false positives out of its own shoulders.
- **Estimating γ**: for exponential decay, $\text{acov}[\text{lag}] \propto \gamma^{\text{lag}}$ where $\gamma = \exp(-1/(\tau \cdot \text{frame\_rate}))$. Fit $\log(\text{acov})$ vs. lag over several lags — one two-point ratio is too noisy for a single real trace.
- **Tune `alpha` (the L1 penalty) by scale.** Too large and the solver returns all zeros; too small and it overfits noise. Real fluorescence (hundreds–thousands of counts) needs a much larger `alpha` than the synthetic data in Deliverable 1 to have a comparable effect.

### What You'll See

**Synthetic validation (Deliverable 1):**

![Synthetic spike recovery](assets/exercise2_synthetic_validation.png)

*Top: noisy vs. noiseless synthetic fluorescence. Middle two rows: true spikes vs. recovered spikes. Bottom: overlay. Recovery is imperfect — some true events are missed, and the recovered trace has spurious small peaks — consistent with F1 ≈ 0.34.*

**Per-cell kernel estimation (part of Deliverable 2):**

![Per-cell tau distribution](assets/exercise2_kernel_estimation.png)

*Left: distribution of estimated τ across 30 real cells, compared to the τ = 1.0s Suite2p used for all of them. Right: example autocovariance curves showing the decay used to fit each cell's τ.*

**Real data comparison (Deliverable 2):**

![Low activity cell comparison](assets/exercise2_real_comparison_low.png)
![Medium activity cell comparison](assets/exercise2_real_comparison_medium.png)
![High activity cell comparison](assets/exercise2_real_comparison_high.png)

*Fluorescence, Suite2p's spike inference, and this exercise's per-cell deconvolution, for a low-, medium-, and high-activity cell. Agreement is generally good on clear events, with some differences in amplitude and on marginal/ambiguous events — consistent with F1 ≈ 0.73.*

---

## Exercise 3: ROI Detection (Challenge)

### The Problem

Before analyzing a neuron, you must **find it** in the raw imaging data. The challenge:

1. **Neurons are small** (~10–15 μm in this dataset, but pixels are ~0.4 μm)
2. **Neuropil is bright** — sometimes brighter than cell bodies
3. **Noise is everywhere** — shot noise, autofluorescence, motion artifacts
4. **Cells overlap** — dendrites cross, tissue is densely packed

### The Existing Solution

Suite2p's default detector (**Sparsery**, [docs](https://suite2p.readthedocs.io/en/latest/roidetection/)) doesn't just look at one static image — it decomposes the whole movie, searching for sources that are spatially compact *and* temporally sparse (active in only a fraction of frames). That's fundamentally more information than a single mean image contains: a real cell's pixels light up and dim together over time in a way that matches its own activity, while bright neuropil regions may be steady, diffuse, or driven by a different (shared, population-wide) timecourse.

**Why not just reimplement Sparsery?** Sparsery is an iterative matrix-decomposition algorithm — a real software engineering project on its own, well beyond what a single exercise can ask of you. This exercise has a different goal: build the *simplest possible* baseline (threshold + connected components) and measure exactly how much you lose by ignoring the movie's temporal structure. Seeing that gap firsthand is what motivates why a more sophisticated, movie-aware method like Sparsery exists in the first place — you don't need to build Sparsery to understand why it's necessary.

In this exercise, you'll build something much simpler — a detector that only looks at **one static image** (the time-averaged fluorescence) — and see how much performance that costs you.

### Deliverable 1 — Detect and Match by Position

**Data**: Use all 125 of Suite2p's detected ROIs (`stat.npy`) as your ground truth, not just the 113 good-quality ones — the benchmark numbers below include the lower-confidence detections too.

Detect cells from the static mean image alone (smooth → threshold → connected components), match your detections against Suite2p's ROI centers by pixel distance, and report sensitivity and precision. Then look at your false positives and false negatives: where do they cluster, and why?

⚠️ Use `ops['meanImg']` (from `ops.npy`) for the image — **not** `F.mean(axis=1)`, which gives one number per already-detected cell, not a spatial image at all.

💬 If your sensitivity/precision look nowhere close to the numbers below (e.g. both near 0%), that's usually a sign something upstream is off — wrong image, wrong coordinate order, or a threshold that's degenerate. Describe your numbers to Cline and it can help you narrow down where.

**Tuning tip**: a loose threshold (e.g. the 80th percentile) lets huge swaths of diffuse neuropil through alongside real cells, and a bare `size > 10 pixels` filter keeps tiny noise specks. Two changes make a real difference without adding any new method: (1) push the threshold much higher — only the brightest ~1% of pixels reliably separates cell bodies from neuropil background in this dataset — and (2) filter components by a size range matching real cells (this dataset's own ROIs span 31–1173 pixels), not just a low floor.

**Expected result** (this dataset):
- With an 80th-percentile threshold and a bare `size > 10` filter: sensitivity **42.4%**, precision **33.5%**
- With a 99th-percentile threshold and a size filter matching real cell dimensions (20–1500 pixels): sensitivity **51.2%**, precision **68.8%** — a substantial improvement from tuning the same simple method, not a different algorithm
- Even tuned, this remains well below Suite2p's Sparsery — a single static image still doesn't contain enough information to separate cells from bright neuropil as reliably as a movie-aware method can

### Deliverable 2 — Does "Matched" Mean "Same Signal"?

Position-matching only checks whether your detection's *center* landed near a Suite2p ROI's center — it says nothing about whether your detected pixels actually capture that cell's activity. A detection could sit right next to a real cell and still count as a "true positive," while actually measuring something else.

**Method**: For every matched pair from Deliverable 1, pull your detection's own raw pixel-averaged trace directly from the movie (`ops.npy`'s registered TIFF, not anything Suite2p precomputed), and correlate it against Suite2p's own `F` trace for that same cell. Then look at a few of your best- and worst-correlated matches side by side — both the image (do the two ROI outlines actually overlap?) and the traces (do they move together?).

**Expected result** (this dataset, using the tuned detector from Deliverable 1):
- Across the 64 matched pairs: mean correlation **0.48**, median **0.43**, range **[0.10, 0.93]**
- 28 of 64 matches (44%) have correlation > 0.5 — genuinely capturing the same signal
- 10 of 64 matches (16%) have correlation < 0.2 — spatially "close enough" but not the same signal
- Better detection (Deliverable 1's tuning) didn't just find more matches — it found *better* ones: both the fraction of high-correlation matches and low-correlation matches improved. Looking at the remaining poor matches, the two ROI outlines are often still visibly non-overlapping — the 30-pixel matching threshold is loose enough to count some clearly-different blobs as a "match"

### What You'll See

![ROI detection comparison](assets/exercise3_roi_detection_results.png)

*Left: the raw mean fluorescence image. Middle: the smoothed threshold mask. Right: your detections (yellow) overlaid on Suite2p's ROI positions (cyan) — note the mismatches in both directions.*

![Match quality: good vs. poor correlation examples](assets/exercise3_match_quality_examples.png)

*Top two rows: "good" matches — the yellow (yours) and cyan (Suite2p) outlines visibly overlap, and the z-scored traces track each other closely. Bottom two rows: "poor" matches — same position-matching criterion counted these as correct, but the outlines are clearly different blobs and the traces don't correlate at all.*

### What You'll Discover

After implementing a simple detector, you'll see:
1. **You miss a substantial fraction of real cells** — a single mean image doesn't separate all cells from background as cleanly as you'd hope
2. **A large share of your detections are false positives** — bright neuropil regions and imaging artifacts pass the same threshold real cells do
3. **Static brightness alone is a weak signal**: some real cells aren't much brighter than their surroundings in the time-averaged image, even though they are clearly active over time
4. **"Correct" isn't binary**: even among your position-matched "true positives," some have essentially no signal agreement with the cell they supposedly matched — sensitivity and precision alone hide this
5. **Why Suite2p's default beats this**: by using the whole movie (not one static image), Suite2p can distinguish sources based on *when* they're active, not just how bright they are on average — information a single-image threshold simply doesn't have access to

This teaches you: **the information you throw away (here, all of the movie's temporal structure) often matters more than the algorithm you use on what's left — and a metric that only checks position can hide exactly how much you're missing.**
