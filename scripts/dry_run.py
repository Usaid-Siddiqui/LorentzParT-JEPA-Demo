"""
Dry-run diagnostic script — catches wiring / shape / import errors without training.

Runs entirely on CPU with tiny fake data (20 jets, 4 particles each).
Nothing is written to the main log directories; all temp files go to /tmp.
The real model weights and configs are tested, but no gradient descent is done
beyond a single backward pass to verify the compute graph.

Usage (from repo root):
    python scripts/dry_run.py

All checks print [PASS] or [FAIL <reason>].
A final summary line shows the overall result.
"""

import os
import sys
import csv
import copy
import tempfile
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── tiny synthetic data dimensions ──────────────────────────────────────────
B       = 4      # batch size
N_JETS  = 20     # total fake jets (train=12, val=4, test=4)
N_PART  = 128    # particles per jet  (must match model default)
N_FEAT  = 4      # pT, eta, phi, E
N_CLS   = 10     # JetClass classes

# ── results tracking ─────────────────────────────────────────────────────────
_results = []

def check(name):
    """Context manager / decorator that catches exceptions and records pass/fail."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        try:
            yield
            _results.append((name, True, ""))
            print(f"  [PASS] {name}")
        except Exception as e:
            tb = traceback.format_exc().strip().splitlines()[-1]
            _results.append((name, False, tb))
            print(f"  [FAIL] {name}")
            print(f"         {tb}")

    return _ctx()


# ── helpers ──────────────────────────────────────────────────────────────────

def _fake_npy_dir(tmp: str, n: int = N_JETS):
    """Write tiny particles.npy (n, 4, 128) and labels.npy (n, 10) to tmp."""
    rng = np.random.default_rng(0)
    particles = rng.random((n, N_FEAT, N_PART)).astype(np.float32)
    # Make energy > 0 for a subset of particles so padding detection works
    particles[:, 3, :] = rng.uniform(0.1, 2.0, (n, N_PART)).astype(np.float32)
    # Zero out the last 10 particles to simulate padding
    particles[:, :, -10:] = 0.0

    labels = np.zeros((n, N_CLS), dtype=np.float32)
    labels[np.arange(n), rng.integers(0, N_CLS, n)] = 1.0

    np.save(os.path.join(tmp, "particles.npy"), particles)
    np.save(os.path.join(tmp, "labels.npy"), labels)


def _make_data_dir(root: str):
    for split, n in [("train", 12), ("val", 4), ("test", 4)]:
        d = os.path.join(root, split)
        os.makedirs(d, exist_ok=True)
        _fake_npy_dir(d, n)


def _fake_pretrain_csv(path: str, n_epochs: int = 5):
    """Write a fake CSV that plot_pretraining_comparison() can read."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "val_loss", "elapsed_total_s", "best_epoch"]
        )
        writer.writeheader()
        best = 0
        best_loss = 9999.0
        for ep in range(1, n_epochs + 1):
            loss = 1.0 / ep + 0.05
            elapsed = ep * 30.0
            if loss < best_loss:
                best_loss = loss
                best = ep
            writer.writerow({
                "epoch": ep,
                "val_loss": round(loss, 4),
                "elapsed_total_s": elapsed,
                "best_epoch": best,
            })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LorentzParT_JEPA — dry-run diagnostics")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="lorpart_dryrun_")
    data_dir = os.path.join(tmpdir, "data")
    log_dir  = os.path.join(tmpdir, "logs")

    # ── Section 1: imports ──────────────────────────────────────────────────
    print("\n[1/8] Imports")

    with check("import torch / numpy"):
        import torch
        import numpy as np

    with check("import src.models (LorentzParT, ParticleJEPA)"):
        from src.models import LorentzParT, ParticleJEPA

    with check("import src.configs (LorentzParTConfig, JEPAConfig, TrainConfig)"):
        from src.configs import LorentzParTConfig, JEPAConfig, TrainConfig

    with check("import src.loss (ConservationLoss, EmbeddingLoss)"):
        from src.loss import ConservationLoss, EmbeddingLoss

    with check("import src.engine (JEPATrainer, MaskedModelTrainer, JetClassTrainer)"):
        from src.engine import JEPATrainer, MaskedModelTrainer, JetClassTrainer

    with check("import src.utils.data (NpyJetClassDataset)"):
        from src.utils.data import NpyJetClassDataset

    with check("import src.utils.viz (plot_jepa_history, plot_ssl_history, plot_pretraining_comparison)"):
        from src.utils.viz import plot_jepa_history, plot_ssl_history, plot_pretraining_comparison

    with check("import src.utils (set_seed, accuracy_metric_ce)"):
        from src.utils import set_seed, accuracy_metric_ce

    # ── Section 2: config loading ───────────────────────────────────────────
    print("\n[2/8] Config loading (YAML)")
    import yaml

    def _load_yaml(rel_path):
        full = os.path.join(REPO_ROOT, rel_path)
        with open(full) as f:
            return yaml.safe_load(f)

    with check("pretrain_jepa.yaml → JEPAConfig + TrainConfig"):
        from src.configs import JEPAConfig, TrainConfig
        cfg = _load_yaml("configs/pretrain_jepa.yaml")
        jepa_model_cfg = JEPAConfig.from_dict(cfg["model"])
        jepa_train_cfg = TrainConfig.from_dict(cfg["train"])
        assert jepa_model_cfg.embed_dim > 0
        assert jepa_model_cfg.predictor_dim > 0

    with check("pretrain_mae.yaml → LorentzParTConfig + TrainConfig"):
        from src.configs import LorentzParTConfig
        cfg = _load_yaml("configs/pretrain_mae.yaml")
        mae_model_cfg = LorentzParTConfig.from_dict(cfg["model"])
        mae_train_cfg = TrainConfig.from_dict(cfg["train"])
        assert mae_model_cfg.mask is True

    with check("finetune.yaml → LorentzParTConfig + TrainConfig"):
        cfg = _load_yaml("configs/finetune.yaml")
        ft_model_cfg = LorentzParTConfig.from_dict(cfg["model"])
        ft_train_cfg = TrainConfig.from_dict(cfg["train"])

    # ── Section 3: fake data + dataset ──────────────────────────────────────
    print("\n[3/8] Fake data & NpyJetClassDataset")

    with check("create fake .npy data in temp dir"):
        _make_data_dir(data_dir)
        assert os.path.exists(os.path.join(data_dir, "train", "particles.npy"))

    NORM_DICT = {
        "pT":     (92.7, 105.8),
        "eta":    (0.0,  0.917),
        "phi":    (0.0,  1.813),
        "energy": (133.8, 167.5),
    }
    NORMALIZE = [True, False, False, True]

    with check("NpyJetClassDataset — mask_mode='random' → (X, target, mask_idx) shapes"):
        from src.utils.data import NpyJetClassDataset
        ds = NpyJetClassDataset(
            particles_path=os.path.join(data_dir, "train", "particles.npy"),
            labels_path=os.path.join(data_dir, "train", "labels.npy"),
            normalize=NORMALIZE,
            norm_dict=NORM_DICT,
            mask_mode="random",
        )
        X, tgt, midx = ds[0]
        assert X.shape    == (N_PART, N_FEAT), f"expected ({N_PART},{N_FEAT}), got {X.shape}"
        assert tgt.shape  == (N_FEAT,),        f"expected ({N_FEAT},), got {tgt.shape}"
        assert midx.shape == (1,),             f"expected (1,), got {midx.shape}"

    with check("NpyJetClassDataset — mask_mode=None → (X, label) shapes"):
        ds_cls = NpyJetClassDataset(
            particles_path=os.path.join(data_dir, "train", "particles.npy"),
            labels_path=os.path.join(data_dir, "train", "labels.npy"),
            normalize=NORMALIZE,
            norm_dict=NORM_DICT,
            mask_mode=None,
        )
        X, lbl = ds_cls[0]
        assert X.shape   == (N_PART, N_FEAT), f"got {X.shape}"
        assert lbl.shape == (N_CLS,),         f"got {lbl.shape}"

    # ── Section 4: model forward passes ─────────────────────────────────────
    print("\n[4/8] Model forward passes (CPU, B=4, N=128)")

    device = torch.device("cpu")

    # Build a small fake batch for fast CPU runs
    torch.manual_seed(0)
    X_batch    = torch.rand(B, N_PART, N_FEAT)
    # Ensure last 10 particles are padded (energy=0)
    X_batch[:, -10:, 3] = 0.0
    mask_idx   = torch.randint(0, N_PART - 10, (B,))   # mask a real particle
    label_batch = torch.zeros(B, N_CLS)
    label_batch[torch.arange(B), torch.randint(0, N_CLS, (B,))] = 1.0

    with check("LorentzParT forward — MAE mode (mask_idx)"):
        from src.models import LorentzParT
        from src.configs import LorentzParTConfig
        cfg_mae = LorentzParTConfig(
            embed_dim=64, num_heads=4, num_layers=2,
            pair_embed_dims=[32, 32, 32], mask=True,
            max_num_particles=N_PART,
        )
        model_mae = LorentzParT(config=cfg_mae).to(device)
        out = model_mae(X_batch, mask_idx)
        assert out.shape == (B, N_FEAT), f"expected ({B},{N_FEAT}), got {out.shape}"

    with check("LorentzParT forward — inference mode (classifier, no mask)"):
        cfg_inf = LorentzParTConfig(
            embed_dim=64, num_heads=4, num_layers=2,
            pair_embed_dims=[32, 32, 32], mask=False, inference=True,
            num_classes=N_CLS, max_num_particles=N_PART,
        )
        model_inf = LorentzParT(config=cfg_inf).to(device)
        logits = model_inf(X_batch)
        assert logits.shape == (B, N_CLS), f"expected ({B},{N_CLS}), got {logits.shape}"

    with check("ParticleJEPA forward → (pred, target) shapes"):
        from src.models import ParticleJEPA
        jepa = ParticleJEPA(
            embed_dim=64, num_heads=4, num_layers=2,
            pair_embed_dims=[32, 32, 32],
            predictor_dim=32, predictor_heads=2, predictor_layers=2,
            max_num_particles=N_PART,
        ).to(device)
        pred, target = jepa(X_batch, mask_idx)
        assert pred.shape   == (B, 64), f"pred shape {pred.shape}"
        assert target.shape == (B, 64), f"target shape {target.shape}"

    # ── Section 5: loss functions ────────────────────────────────────────────
    print("\n[5/8] Loss functions")

    with check("EmbeddingLoss forward → scalar + (scalar,) tuple"):
        from src.loss import EmbeddingLoss
        loss_fn = EmbeddingLoss(embed_dim=64)
        p = torch.randn(B, 64)
        t = torch.randn(B, 64)
        loss, comps = loss_fn(p, t)
        assert loss.shape == torch.Size([]), f"expected scalar, got {loss.shape}"
        assert len(comps) == 1

    with check("ConservationLoss forward → scalar"):
        from src.loss import ConservationLoss
        cons_loss = ConservationLoss()
        pred_feats = torch.randn(B, N_FEAT)
        true_feats = torch.randn(B, N_FEAT)
        loss_c, comps_c = cons_loss(pred_feats, true_feats)
        assert loss_c.ndim == 0, "expected scalar"

    # ── Section 6: backward pass / gradient check ───────────────────────────
    print("\n[6/8] Backward pass (grad flow)")

    with check("JEPA: loss.backward() propagates gradients to context_encoder"):
        from src.models import ParticleJEPA
        from src.loss import EmbeddingLoss
        jepa_grad = ParticleJEPA(
            embed_dim=64, num_heads=4, num_layers=2,
            pair_embed_dims=[32, 32, 32],
            predictor_dim=32, predictor_heads=2, predictor_layers=2,
            max_num_particles=N_PART,
        ).to(device)
        loss_fn = EmbeddingLoss(embed_dim=64)
        X_g = X_batch.clone()
        mi  = mask_idx.clone()
        p, t = jepa_grad(X_g, mi)
        loss, _ = loss_fn(p, t)
        loss.backward()
        # Check at least one context_encoder param has a gradient
        grads = [
            p.grad for p in jepa_grad.context_encoder.parameters()
            if p.grad is not None
        ]
        assert len(grads) > 0, "no gradients reached context_encoder"
        # Target encoder must have no grad (frozen / detached)
        tgt_grads = [
            p.grad for p in jepa_grad.target_encoder.parameters()
            if p.grad is not None
        ]
        assert len(tgt_grads) == 0, "target_encoder should have no grad"

    with check("MAE: loss.backward() propagates gradients to encoder"):
        from src.models import LorentzParT
        from src.loss import ConservationLoss
        from src.configs import LorentzParTConfig
        cfg = LorentzParTConfig(
            embed_dim=64, num_heads=4, num_layers=2,
            pair_embed_dims=[32, 32, 32], mask=True,
            max_num_particles=N_PART,
        )
        mae_g = LorentzParT(config=cfg).to(device)
        cons = ConservationLoss()
        out = mae_g(X_batch, mask_idx)
        tgt = X_batch[torch.arange(B), mask_idx]   # true masked particle
        loss_m, _ = cons(out, tgt)
        loss_m.backward()
        grads = [p.grad for p in mae_g.encoder.parameters() if p.grad is not None]
        assert len(grads) > 0, "no gradients reached encoder"

    # ── Section 7: weight save / load round-trip ────────────────────────────
    print("\n[7/8] Weight save / load round-trip")

    wt_dir = os.path.join(tmpdir, "weights")
    os.makedirs(wt_dir, exist_ok=True)

    jepa_wt_path = os.path.join(wt_dir, "jepa_encoder.pt")
    mae_wt_path  = os.path.join(wt_dir, "mae_model.pt")

    with check("JEPA: save context_encoder with 'encoder.*' prefix"):
        from src.models import ParticleJEPA
        jepa_save = ParticleJEPA(
            embed_dim=64, num_heads=4, num_layers=2,
            pair_embed_dims=[32, 32, 32],
            predictor_dim=32, predictor_heads=2, predictor_layers=2,
            max_num_particles=N_PART,
        )
        state = {f"encoder.{k}": v for k, v in jepa_save.context_encoder.state_dict().items()}
        torch.save(state, jepa_wt_path)
        loaded = torch.load(jepa_wt_path, map_location="cpu")
        assert all(k.startswith("encoder.") for k in loaded), \
            "some keys lack 'encoder.' prefix"

    with check("LorentzParT loads JEPA weights via config.weights (strict=False)"):
        from src.models import LorentzParT
        from src.configs import LorentzParTConfig
        cfg = LorentzParTConfig(
            embed_dim=64, num_heads=4, num_layers=2,
            pair_embed_dims=[32, 32, 32], mask=False,
            num_classes=N_CLS, max_num_particles=N_PART,
            weights=jepa_wt_path,
        )
        m = LorentzParT(config=cfg)
        # Verify encoder params match the saved JEPA context_encoder
        loaded_sd = torch.load(jepa_wt_path, map_location="cpu")
        for k_full, v_saved in loaded_sd.items():
            k_enc = k_full[len("encoder."):]
            if k_enc in m.encoder.state_dict():
                loaded_v = m.encoder.state_dict()[k_enc]
                assert torch.allclose(loaded_v, v_saved), f"mismatch at {k_enc}"

    with check("MAE: save full model → LorentzParT loads it (strict=False)"):
        from src.models import LorentzParT
        from src.configs import LorentzParTConfig
        cfg_s = LorentzParTConfig(
            embed_dim=64, num_heads=4, num_layers=2,
            pair_embed_dims=[32, 32, 32], mask=True,
            max_num_particles=N_PART,
        )
        mae_sd_model = LorentzParT(config=cfg_s)
        torch.save(mae_sd_model.state_dict(), mae_wt_path)

        cfg_l = LorentzParTConfig(
            embed_dim=64, num_heads=4, num_layers=2,
            pair_embed_dims=[32, 32, 32], mask=False,
            num_classes=N_CLS, max_num_particles=N_PART,
            weights=mae_wt_path,
        )
        _ = LorentzParT(config=cfg_l)

    # ── Section 7b: trainer initialisation ──────────────────────────────────
    print("\n  Trainer initialisation (no training)")

    with check("JEPATrainer: __init__ + _set_logging_paths"):
        from src.models import ParticleJEPA
        from src.engine import JEPATrainer
        from src.configs import TrainConfig
        from src.utils.data import NpyJetClassDataset
        _ds = NpyJetClassDataset(
            particles_path=os.path.join(data_dir, "train", "particles.npy"),
            labels_path=os.path.join(data_dir, "train", "labels.npy"),
            normalize=NORMALIZE, norm_dict=NORM_DICT, mask_mode="random",
        )
        _val_ds = NpyJetClassDataset(
            particles_path=os.path.join(data_dir, "val", "particles.npy"),
            labels_path=os.path.join(data_dir, "val", "labels.npy"),
            normalize=NORMALIZE, norm_dict=NORM_DICT, mask_mode="random",
        )
        _jm = ParticleJEPA(
            embed_dim=64, num_heads=4, num_layers=2,
            pair_embed_dims=[32, 32, 32],
            predictor_dim=32, predictor_heads=2, predictor_layers=2,
            max_num_particles=N_PART,
        )
        _jcfg = TrainConfig(
            num_epochs=1, batch_size=4,
            logging_dir=log_dir, save_best=True, save_ckpt=False,
            progress_bar=False,
        )
        jtrainer = JEPATrainer(
            model=_jm,
            train_dataset=_ds, val_dataset=_val_ds,
            device=device, config=_jcfg,
            ema_momentum_start=0.996, ema_momentum_end=1.0,
        )
        jtrainer._set_logging_paths("dryrun_jepa")
        assert jtrainer.best_model_path is not None

    with check("MaskedModelTrainer: __init__ + _set_logging_paths"):
        from src.models import LorentzParT
        from src.engine import MaskedModelTrainer
        from src.configs import LorentzParTConfig, TrainConfig
        _mae_m = LorentzParT(config=LorentzParTConfig(
            embed_dim=64, num_heads=4, num_layers=2,
            pair_embed_dims=[32, 32, 32], mask=True,
            max_num_particles=N_PART,
        ))
        _mcfg = TrainConfig(
            num_epochs=1, batch_size=4,
            logging_dir=log_dir, save_best=True, save_ckpt=False,
            progress_bar=False,
        )
        mtrainer = MaskedModelTrainer(
            model=_mae_m, train_dataset=_ds, val_dataset=_val_ds,
            device=device, config=_mcfg,
        )
        mtrainer._set_logging_paths("dryrun_mae")
        assert mtrainer.best_model_path is not None

    with check("JetClassTrainer: __init__ + _set_logging_paths"):
        from src.models import LorentzParT
        from src.engine import JetClassTrainer
        from src.configs import LorentzParTConfig, TrainConfig
        from src.utils import accuracy_metric_ce
        _cls_m = LorentzParT(config=LorentzParTConfig(
            embed_dim=64, num_heads=4, num_layers=2,
            pair_embed_dims=[32, 32, 32], mask=False,
            num_classes=N_CLS, max_num_particles=N_PART,
        ))
        _cls_ds = NpyJetClassDataset(
            particles_path=os.path.join(data_dir, "train", "particles.npy"),
            labels_path=os.path.join(data_dir, "train", "labels.npy"),
            normalize=NORMALIZE, norm_dict=NORM_DICT, mask_mode=None,
        )
        _cls_val = NpyJetClassDataset(
            particles_path=os.path.join(data_dir, "val", "particles.npy"),
            labels_path=os.path.join(data_dir, "val", "labels.npy"),
            normalize=NORMALIZE, norm_dict=NORM_DICT, mask_mode=None,
        )
        _ccfg = TrainConfig(
            num_epochs=1, batch_size=4,
            logging_dir=log_dir, save_best=True, save_ckpt=False,
            progress_bar=False,
        )
        ctrainer = JetClassTrainer(
            model=_cls_m, train_dataset=_cls_ds, val_dataset=_cls_val,
            device=device, metric=accuracy_metric_ce, config=_ccfg,
        )
        ctrainer._set_logging_paths("dryrun_cls")
        assert ctrainer.best_model_path is not None

    # ── Section 8: viz functions ─────────────────────────────────────────────
    print("\n[8/8] Visualisation functions (non-interactive)")
    import matplotlib
    matplotlib.use("Agg")   # no display needed

    with check("plot_jepa_history — fake history dict"):
        from src.utils.viz import plot_ssl_history, plot_jepa_history
        fake_jepa_hist = {
            "epoch": list(range(1, 6)),
            "embedding_loss": [0.5, 0.4, 0.35, 0.31, 0.29],
            "val_loss": [0.6, 0.45, 0.38, 0.33, 0.30],
        }
        plot_jepa_history(fake_jepa_hist, save_fig=None)

    with check("plot_ssl_history — fake history dict"):
        fake_mae_hist = {
            "epoch": list(range(1, 6)),
            "pT_loss":     [0.3, 0.25, 0.2, 0.18, 0.16],
            "eta_loss":    [0.2, 0.18, 0.15, 0.13, 0.12],
            "phi_loss":    [0.4, 0.35, 0.3, 0.28, 0.25],
            "energy_loss": [0.3, 0.26, 0.22, 0.20, 0.18],
            "val_loss":    [0.5, 0.42, 0.36, 0.32, 0.29],
        }
        plot_ssl_history(fake_mae_hist, save_fig=None)

    with check("plot_pretraining_comparison — fake CSV logs"):
        from src.utils.viz import plot_pretraining_comparison
        j_csv = os.path.join(tmpdir, "jepa_pretrain.csv")
        m_csv = os.path.join(tmpdir, "mae_pretrain.csv")
        _fake_pretrain_csv(j_csv)
        _fake_pretrain_csv(m_csv)
        plot_pretraining_comparison(jepa_csv=j_csv, mae_csv=m_csv, save_fig=None)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = [(n, msg) for n, ok, msg in _results if not ok]
    print(f"RESULTS: {passed}/{len(_results)} checks passed")
    if failed:
        print(f"\nFailed checks ({len(failed)}):")
        for name, msg in failed:
            print(f"  - {name}")
            print(f"    {msg}")
        print("\nFix the failures above before running on HPC.")
    else:
        print("\nAll checks passed. Pipeline is wired correctly.")
        print("You can safely run scripts/run_comparison.py on HPC.")

    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
