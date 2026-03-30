# LorentzParT-JEPA

**GSoC 2026 — Event Classification with Masked Transformer Autoencoders**

This repository implements a **JEPA (Joint-Embedding Predictive Architecture)** pretraining
pipeline for the LorentzParT model, extending last year's masked autoencoder (MAE) work with
a latent-space prediction objective.

---

## Overview

Both approaches mask one particle per jet and use it as the pretraining signal.
The key difference is *what* gets predicted:

| Approach | Predicts | Loss |
|----------|----------|------|
| **MAE (baseline)** | Raw 4-vector (pT, η, φ, E) | ConservationLoss (√MSE + cosine for φ) |
| **JEPA (this work)** | Target encoder's latent embedding | MSE after LayerNorm (collapse-resistant) |

JEPA learns in embedding space — it avoids memorising low-level detector noise and
focuses on the physics relationships between particles instead.

---

## Architecture

```
ParticleJEPA (pretraining)
├── processor        : (pT, η, φ, E) → 16-dim Lorentz multivectors + pairwise U
├── context_encoder  : LorentzParTEncoder on masked input  →  (B, 128, 128)
│     └── EquiLinear(1→1) → Linear(16→128) → 8× ParticleAttentionBlock
├── target_encoder   : EMA copy of context_encoder, sees full unmasked input
└── predictor        : bottleneck transformer (64-dim, 4 heads, 4 layers)
      Linear(128→64) → positional embed → mask token → 4× TransformerBlock
      → Linear(64→128)  →  predicted embedding (B, 128)

EMA update:  θ_target ← m·θ_target + (1−m)·θ_context,  m: 0.996 → 1.0

LorentzParT (fine-tuning / from scratch)
├── encoder  : LorentzParTEncoder (loaded from context_encoder weights)
└── decoder  : learnable CLS token → 2× ClassAttentionBlock → Linear(128→10)
```

---

## Dataset

JetClass 100k balanced subset (10,000 jets per class):

| Class | Process |
|-------|---------|
| 0 | Z→νν (background QCD) |
| 1 | H→bb̄ |
| 2 | H→cc̄ |
| 3 | H→gg |
| 4 | H→4q |
| 5 | H→ℓνqq′ |
| 6 | Z→qq̄ |
| 7 | W→qq′ |
| 8 | t→bqq′ |
| 9 | t→bℓν |

Split: **80k train / 10k val / 10k test** (balanced, deterministic seed).

---

## Installation

```bash
# Clone and enter the repo
cd LorentzParT_JEPA

# Install dependencies
pip install -r requirements.txt

# Install L-GATr (Lorentz-equivariant components)
pip install git+https://github.com/heidelberg-hepml/lorentz-gatr.git
```

---

## Reproducing the Experiments

### 1 — Prepare Data

Extract 100k events from the `val_5M` ROOT files:

```bash
python scripts/prepare_data.py \
    --data-dir /path/to/val_5M \
    --output-dir ./data \
    --seed 42
```

Output: `./data/{train,val,test}/{particles.npy, labels.npy}`

### 2 — JEPA Pretraining

```bash
python scripts/pretrain_jepa.py \
    --data-dir ./data \
    --config-path configs/pretrain_jepa.yaml \
    --seed 42
```

Best model saved to `./logs/ParticleJEPA/best/<run>.pt`

### 3 — MAE Pretraining (Baseline)

```bash
python scripts/pretrain_mae.py \
    --data-dir ./data \
    --config-path configs/pretrain_mae.yaml \
    --seed 42
```

Best model saved to `./logs/LorentzParT/best/<run>.pt`

### 4 — Fine-tuning

```bash
# JEPA pretrained
python scripts/finetune.py \
    --data-dir ./data \
    --weights ./logs/ParticleJEPA/best/<run>.pt \
    --run-name jepa_finetune \
    --seed 42

# MAE pretrained
python scripts/finetune.py \
    --data-dir ./data \
    --weights ./logs/LorentzParT/best/<run>.pt \
    --run-name mae_finetune \
    --seed 42

# From scratch
python scripts/finetune.py \
    --data-dir ./data \
    --run-name scratch \
    --seed 42
```

### 5 — Evaluation

```bash
python scripts/evaluate.py \
    --data-dir ./data \
    --weights ./logs/LorentzParT/best/jepa_finetune.pt \
    --run-name jepa_finetune
```

### Run Everything at Once

```bash
python scripts/run_comparison.py --data-dir ./data --seed 42
```

### Interactive Notebook

```bash
jupyter notebook demo.ipynb
```

---

## Multi-GPU / HPC

All scripts support multi-GPU via PyTorch DDP. On a SLURM cluster:

```bash
# Single node, all GPUs
python scripts/pretrain_jepa.py --data-dir ./data

# Or via torchrun
torchrun --nproc_per_node=4 scripts/pretrain_jepa.py --data-dir ./data
```

The scripts automatically detect `torch.cuda.device_count()` and spawn accordingly.

---

## Expected Results (100k subset)

On 100k samples the absolute accuracy numbers are lower than on the full 100M dataset,
but the relative ordering should hold:

| Condition | Expected accuracy (approx) |
|-----------|---------------------------|
| JEPA pretrained | ~60–65% |
| MAE pretrained  | ~58–63% |
| From scratch    | ~55–60% |

The value of JEPA is most apparent at limited data budgets and with longer pretraining
on the full dataset. The proposal argues this approach scales better.

---

## File Structure

```
LorentzParT_JEPA/
├── README.md
├── demo.ipynb                     # Interactive notebook
├── requirements.txt
├── configs/
│   ├── pretrain_jepa.yaml         # JEPA pretraining config
│   ├── pretrain_mae.yaml          # MAE pretraining config
│   ├── finetune.yaml              # Shared fine-tuning config
│   └── evaluate.yaml
├── scripts/
│   ├── prepare_data.py            # Extract 100k subset
│   ├── pretrain_jepa.py           # JEPA pretraining entry point
│   ├── pretrain_mae.py            # MAE pretraining entry point
│   ├── finetune.py                # Fine-tuning (all 3 conditions)
│   ├── evaluate.py                # Test-set evaluation + plots
│   └── run_comparison.py          # Orchestrate all experiments
└── src/
    ├── models/
    │   ├── jepa.py                # ParticleJEPA (NEW)
    │   ├── predictor.py           # ParticlePredictor (NEW)
    │   ├── lorentz_part.py        # LorentzParT (copied)
    │   ├── processor.py           # ParticleProcessor (copied)
    │   └── ...
    ├── engine/
    │   ├── jepa_trainer.py        # JEPATrainer (NEW)
    │   ├── mm_trainer.py          # MaskedModelTrainer (MAE, copied)
    │   ├── jetclass_trainer.py    # Classification trainer (copied)
    │   └── ...
    ├── loss/
    │   ├── embedding_loss.py      # EmbeddingLoss (NEW)
    │   ├── conservation_loss.py   # ConservationLoss (copied)
    │   └── ...
    └── utils/data/
        ├── jetclass.py            # NpyJetClassDataset (NEW) + existing classes
        └── ...
```

---

## References

1. Qu, H. et al. "Particle Transformer for Jet Tagging." *ICML 2022*.
2. Spinner, J. et al. "Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics." *NeurIPS 2024*.
3. Assran, M. et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." *CVPR 2023*.
4. Nguyen, T.P. "GSoC 2025: LorentzParT Hybrid Model." *Medium 2025*.
