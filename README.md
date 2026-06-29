# Activation Probe: Truthfulness vs. Hallucination in Llama 3.2 3B

Linear probing and control vector steering to find where — and how strongly — Llama 3.2 3B encodes the difference between truthful and hallucinated statements in its residual stream.

## What it does

1. Constructs a dataset of 20 matched pairs (factual vs. hallucinated statements, identical structure)
2. Extracts residual stream activations at every layer for each statement
3. Trains a linear probe (logistic regression, 5-fold CV) per layer to classify truthful vs. hallucinated
4. Plots probe accuracy by layer to identify where the model linearly encodes the distinction
5. Computes a control vector (mean difference in activation space at the peak layer)
6. Applies the control vector at inference time via a forward hook to steer generation
7. Shows before/after generation examples at multiple steering strengths

## Running in Google Colab (recommended)

The notebook requires a GPU. Free Colab T4 is sufficient.

**Step 1 — Open in Colab**

Go to [colab.research.google.com](https://colab.research.google.com), click **File > Open notebook > GitHub**, and paste this repo URL.

**Step 2 — Enable GPU**

In Colab: **Runtime > Change runtime type > Hardware accelerator: T4 GPU > Save**

**Step 3 — HuggingFace token**

Llama 3.2 is a gated model. You need to:
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the license at [huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
3. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is enough)

The notebook calls `login()` which will prompt you to paste the token.

**Step 4 — Run all cells**

Runtime > Run all. Full run takes roughly 10-15 minutes on a T4.

## Running locally

Requires ~8GB VRAM or ~16GB RAM (CPU is slow but works).

```bash
pip install transformers accelerate scikit-learn matplotlib seaborn torch
jupyter notebook activation_probe.ipynb
```

## Key outputs

- `probe_accuracy.png` — probe accuracy by layer + accuracy by network depth region
- `control_vector.png` — projection distributions and per-pair separation scores
- Console summary with peak layer, Cohen d separation, and correctly ordered pairs

## Related work

- Zou et al., 2023 — [Representation Engineering](https://arxiv.org/abs/2310.01405)
- Marks & Tegmark, 2023 — [The Geometry of Truth](https://arxiv.org/abs/2310.06824)
- Burns et al., 2022 — [Discovering Latent Knowledge](https://arxiv.org/abs/2212.03827)
