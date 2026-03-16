# Neural Language Models From Scratch

Implemented series of language models from scratch — from bigram frequency tables to a character-level GPT — with a deep dive into BatchNorm and attention mechanisms

Referenced  Andrej Karpathy's makemore series 

---

## Overview

This project builds a series of increasingly powerful character-level language models, each motivated by the weakness of the previous one. The simplest models are trained to generate names by predicting the next character, while the most advanced one (Transformer) generates Shakespeare like text autoregressively. 

| Notebook | Model | Val Loss | Key Concepts |
|---|---|---|---|
| `01_bigram.ipynb` | Bigram (freq table + 1-layer NN) | ~2.57 | MLE, one-hot encoding, negative log-likelihood |
| `02_mlp.ipynb` | MLP (Bengio 2003) | ~2.32 | Embeddings, context windows, learning rate search |
| `03_mlp_batchnorm.ipynb` | MLP + BatchNorm | ~2.11 | Kaiming init, dead neurons, activation diagnostics |
| `04_wavenet.ipynb` | WaveNet-style hierarchical MLP | ~2.02 | Dilated convolutions (conceptual), hierarchical feature fusion |
| `05_gpt.ipynb` | Character-level GPT | ~1.76 | Self-attention, multi-head attention, residual + LayerNorm |

---

## Progression

```
Bigram → MLP → MLP + BatchNorm → WaveNet → GPT
  ↑                                          ↑
Simple frequency counts              Full transformer decoder
  (no learning beyond pairs)         (attention across full context)
```

Each step introduces one key idea:

1. **Bigram**: Frequency table of character pairs in data, converted into a probability table. Given any character, sample from probability distribution to get next character
1. **MLP**: Move from statistic-based method to neural net, enabling (1) larger context window (>1 input characters) and (2) learned embeddings to generalise to unseen character pairs.
2. **MLP + BatchNorm**: Fix high initial loss, diagnose saturated `tanh` activations and dead neurons using visualisation tools. Introduce Kaiming initialisation
3. **WaveNet**: Replace the flat context-concatenation with a hierarchical structure that preserves positional order
4. **GPT**: Replace fixed-context MLP with self-attention. Tokens dynamically aggregate information from all prior tokens, weighted by learned query-key affinities.

--- 

**Key features of project:**
- Diagnostic visualisations explaining why each architectural improvement exists and its impact (highlights: `03_mlp_batchnorm.ipynb` plots on gradients and activation distributions pre and post normalisation)
- Implementation of transformer architecture
- Implementation of neural network layers from scratch in `02_mlp.ipynb` and `03_mlp_batchnorm.ipynb` without importing libraries from PyTorch

---

### Neural Network Concepts
- Character-level tokenisation and vocabulary construction
- Embedding tables (learned lookup)
- Kaiming (He) initialisation
- Batch Normalisation: forward pass, running stats, inference mode
- Layer Normalisation (used in transformers)
- Residual connections and why they help gradient flow

### Transformer architecture
  - Token and positional embeddings
  - **3 Transformer Blocks** (each with 2 attention heads + feedforward + residual connection + layer norm) 
  - Linear layer + Softmax Layer

Slides: [`Building Names Generator - MLP.pdf`](./slides/Building%20Names%20Generator%20-%20MLP.pdf)  
Slides: [`Building GPT.pdf`](./slides/Building%20GPT.pdf)  

---

## Results

### Names Generator — Loss Progression

| Model | Train Loss | Val Loss |
|---|---|---|
| Bigram (statistical) | — | ~2.57 |
| Bigram (neural, 1-layer) | ~2.57 | ~2.57 |
| MLP | ~2.25 | ~2.32 |
| MLP + BatchNorm | ~2.07 | ~2.11 |
| WaveNet | ~1.92 | ~2.02 |

### Tiny GPT
- **Parameters**: 158,913 (~159K)
- **Final train loss**: ~1.68 | **Val loss**: ~1.76 (after 15k steps)
- **Architecture**: 3 layers, 2 heads, 64-dim embeddings, context length 16

Sample output after training:
```
this fast,
And livted with eylard,
And what sreft,
By this me this knows
Beaves my trunk.
```

### Names samples (WaveNet)
```
sanay, zuham, zaison, agondre, karina, joberlee
```

---

## Design Decisions & Notes

**Why character-level?** Character-level models require no tokeniser, keep vocabulary tiny (27 tokens), and make the full pipeline transparent — every piece from data loading to generation is visible and debuggable.

**Why names generator first, then Shakespeare?** Names give a short feedback loop: you can see whether generated outputs look plausible immediately (are they pronounceable?). Shakespeare then demonstrates the same attention mechanisms work on real language.

---


## Running the Notebooks

```bash
git clone https://github.com/<your-username>/neural-lm-from-scratch
cd neural-lm-from-scratch
pip install -r requirements.txt
jupyter notebook notebooks/
```

**Requirements**: Python 3.9+, PyTorch 2.0+, matplotlib. No GPU needed for notebooks 1–5; notebook 6 benefits from a GPU for the full 15k-iteration run.

---

## References

- Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) and [nanoGPT](https://github.com/karpathy/nanoGPT) series  
- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) — Bengio et al. 2003
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) — van den Oord et al. 2016
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. 2017
- [Batch Normalization](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy 2015
