"""
Evaluation script — measures classification performance on the test set.

Produces:
  - Accuracy and per-class accuracy
  - ROC curves (one-vs-rest)
  - Confusion matrix

Usage:
    python scripts/evaluate.py \\
        --data-dir ./data \\
        --weights ./logs/LorentzParT/best/jepa_finetune.pt \\
        --run-name jepa_eval
"""

import os
import sys
import yaml
import argparse
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs import LorentzParTConfig, TrainConfig
from src.engine import JetClassTrainer
from src.models import LorentzParT
from src.utils import accuracy_metric_ce, set_seed
from src.utils.data import NpyJetClassDataset
from src.utils.viz import plot_roc_curve, plot_confusion_matrix

warnings.filterwarnings('ignore')


NORM_DICT = {
    'pT':     (92.72917175292969,  105.83937072753906),
    'eta':    (0.0005733045982196927, 0.9174848794937134),
    'phi':    (-0.00041169871110469103, 1.8136887550354004),
    'energy': (133.8745574951172, 167.528564453125),
}
NORMALIZE = [True, False, False, True]

CLASS_NAMES = [
    'QCD/Z→νν', 'H→bb̄', 'H→cc̄', 'H→gg', 'H→4q',
    'H→ℓνqq′', 'Z→qq̄', 'W→qq′', 't→bqq′', 't→bℓν'
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LorentzParT classifier")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--config-path', type=str, default='./configs/finetune.yaml')
    parser.add_argument('--weights', type=str, required=True,
                        help="Path to fine-tuned model weights (.pt)")
    parser.add_argument('--run-name', type=str, default='eval')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outputs-dir', type=str, default='./outputs')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = LorentzParTConfig.from_dict(config['model'])
    train_cfg = TrainConfig.from_dict(config['train'])
    model_cfg.inference = True

    test_dataset = NpyJetClassDataset(
        particles_path=os.path.join(args.data_dir, 'test', 'particles.npy'),
        labels_path=os.path.join(args.data_dir, 'test', 'labels.npy'),
        normalize=NORMALIZE,
        norm_dict=NORM_DICT,
        mask_mode=None,
    )

    # Dummy train/val datasets (not used for evaluation but required by Trainer)
    train_dataset = NpyJetClassDataset(
        particles_path=os.path.join(args.data_dir, 'train', 'particles.npy'),
        labels_path=os.path.join(args.data_dir, 'train', 'labels.npy'),
        normalize=NORMALIZE,
        norm_dict=NORM_DICT,
        mask_mode=None,
    )

    # Load model with fine-tuned weights (full model, not just encoder)
    model = LorentzParT(config=model_cfg)
    state_dict = torch.load(args.weights, map_location='cpu')
    # If saved as full model state_dict (from finetune.py), load directly
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    trainer = JetClassTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        test_dataset=test_dataset,
        device=device,
        metric=accuracy_metric_ce,
        config=train_cfg,
    )
    trainer._set_logging_paths(args.run_name)

    # Override outputs_dir so plots go to the user-specified --outputs-dir,
    # not the Trainer's default ./logs/LorentzParT/output/
    os.makedirs(args.outputs_dir, exist_ok=True)
    trainer.outputs_dir = args.outputs_dir

    test_loss, test_acc, y_true, y_pred = trainer.evaluate(
        loss_type='cross_entropy',
        plot=[plot_roc_curve, plot_confusion_matrix]
    )

    print(f"\n{'=' * 50}")
    print(f"Run:      {args.run_name}")
    print(f"Weights:  {args.weights}")
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"{'=' * 50}")

    # Per-class accuracy
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    print("\nPer-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true_labels == i
        if mask.sum() == 0:
            continue
        acc_i = (y_pred_labels[mask] == i).mean()
        print(f"  Class {i:2d} ({name:12s}): {acc_i:.4f}")

    return test_loss, test_acc


if __name__ == '__main__':
    main()
