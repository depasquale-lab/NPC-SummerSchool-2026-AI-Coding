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

## Finished Early, or Want to Go Further?

Your Claude Code access isn't limited to these two sessions — it's active for the **duration of the whole summer school**. If you finish Friday and Saturday's exercises (or just want to branch off from them), you're welcome to point Claude Code at your **own data** instead of `TSeries-03042024-run02-054`: the same techniques (neuropil correction, spike deconvolution, ROI detection, PCA, spike-triggered averaging) generalize to any Suite2p-processed two-photon recording, not just this one. Ask Claude Code to help you adapt the exercise notebooks to your own file paths and data shapes — that adaptation exercise, done thoughtfully, is genuinely useful practice in its own right, not just busywork before "the real thing."

## Claude Code Crash Course: Usage, Models, and Compacting

A few practical things worth knowing about running Claude Code itself, distinct from anything about the neuroscience:

**Checking your usage.** Type `/usage` in the Claude Code chat at any point to see how close you are to your rate limit. If you're ever unsure whether you can afford to keep going on something, checking is free — it doesn't cost you anything to look.

**Switching models — the speed/quality tradeoff.** Type `/model` to see and switch between available models. Claude Code defaults to a strong, capable model, but you can switch to a smaller, faster one for simpler tasks. The tradeoff is real in both directions: a smaller model uses up your usage allowance more slowly (so you get more total requests before hitting a limit), but it's also noticeably worse at hard, multi-step reasoning — the kind of thing several of these exercises actually require (e.g. debugging the causal-convolution alignment issue in Exercise 2, or reasoning through why a metric's matching rule matters in Exercise 3). A reasonable default: stay on the capable model for anything conceptually hard or unfamiliar, and only switch down for genuinely simple, mechanical requests (renaming a variable, adding a print statement) if you're trying to conserve usage.

**What "compacting" means.** As a conversation with Claude Code gets long, it eventually fills up the model's available context window (the amount of text it can actually "see" at once, including everything said so far). When that happens, Claude Code automatically **compacts** — it summarizes the earlier parts of the conversation into a shorter form that preserves the important decisions and context, freeing up room to keep working instead of hitting a hard wall. You can also trigger this yourself with `/compact` if a session has gotten long and sluggish. This is normal and expected on longer sessions (like working through a whole exercise in one sitting) — it's not a sign anything went wrong, just Claude Code managing its own memory so the conversation can keep going.
