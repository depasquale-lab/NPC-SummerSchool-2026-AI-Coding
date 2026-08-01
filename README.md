# Calcium Imaging Data Processing with Agentic AI

❗ **Already working on your own branch from an earlier session?** `main` has changed substantially since then — sync before you continue, or you'll be working from a stale copy of these documents. ("Merging" just means folding the new changes from `main` into your branch, so your branch ends up with both your own work *and* everyone else's updates since — usually automatic, but if the same lines were changed in both places, you'll be asked to help decide which version wins.) Make sure you're on your own branch (`git branch` should star your branch, not `main`), then ask Claude Code: "Fetch the latest changes from origin, then merge main into my current branch." Have it walk you through any merge conflicts rather than guessing at them yourself.

## The Big Picture

Over two sessions, you'll use Claude Code to analyze a real two-photon calcium imaging recording from an awake mouse — going from raw, noisy fluorescence video all the way to population-level structure across the recorded neurons. Every step follows the same pattern: understand what a real pipeline (Suite2p) does, build a simpler version of it yourself with Claude Code's help, and validate your result against a real reference before trusting it.

This isn't just a neuroscience exercise. It's also about learning to work *with* an AI coding agent — describing what you want, reading its code well enough to sanity-check it, catching it when it's wrong, and verifying every real result against something independent, rather than trusting a number just because code produced it.

## Which Document to Use

**➡️ Start with [`FRIDAY_EXERCISES.md`](FRIDAY_EXERCISES.md).** This covers setup (SCC access, Claude Code, cloning this repo, Python environment) and the three core exercises:

1. **Neuropil Removal** (Warmup) — removing contamination from surrounding tissue
2. **Spike Deconvolution** (Main) — recovering spike times from slow calcium dynamics
3. **ROI Detection** (Challenge) — finding neurons in a raw image in the first place

**➡️ Then move to [`SATURDAY_EXERCISES.md`](SATURDAY_EXERCISES.md).** This assumes you've completed Friday's setup and Exercise 1, and moves from single-cell analysis to the whole recorded population:

4. **PCA** — does population activity live in a low-dimensional subspace, or does it need most of its dimensions?
5. **Spike-Triggered Averaging** — when one neuron fires, do others respond?

Both documents are self-contained guides with their own deliverables, expected results, and tips for working with Claude Code — read whichever one applies to where you are, rather than this file.

## Everyone Uses the Same Dataset

`TSeries-03042024-run02-054` — every number, benchmark, and figure in both documents was generated from this exact recording. Using a different dataset is fine if you're exploring independently, but the "results should look something like this" sections won't apply.
