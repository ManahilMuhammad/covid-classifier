"""
evaluate trained checkpoint using:
1. metrics
2. confusion matrix
3. ROC curves
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score, 
    classification_report,
    confusion_matrix, 
    log_loss, 
    roc_auc_score, 
    roc_curve
)
from tqdm import tqdm

from . import plotting as P
from .config import ASSETS_DIR, CHECKPOINT, CLASS_LABELS, CLASSES, RESULTS_DIR
from .data import ChestXRayDataset
from .model import get_device, load_model


@torch.no_grad()
def predict(model, loader, device, desc="predict"):
    """
    return probabilities and labels for every sample in the loader
    """
    model.eval()
    probs, labels = [], []

    # get probable label for each sample along with its actual label,
    for images, y in tqdm(loader, desc=desc, leave=False):
        probs.append(torch.softmax(model(images.to(device)), dim=1).cpu())
        labels.append(y)

    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def compute_metrics(probs, labels):
    preds = probs.argmax(1)

    # generate report using actual and predicted labels
    report = classification_report(
        labels, preds, 
        labels=range(len(CLASSES)),
        target_names=CLASSES, 
        output_dict=True, zero_division=0
    )

    # return metrics including per-class metrics
    return {
        'n': int(len(labels)),
        'loss': float(log_loss(labels, probs, labels=range(len(CLASSES)))),
        'accuracy': float((preds == labels).mean()),
        'balanced_accuracy': float(balanced_accuracy_score(labels, preds)),
        'macro_f1': float(report['macro avg']['f1-score']),
        'per_class': {
            c: {
                'precision': float(report[c]['precision']),
                'recall': float(report[c]['recall']),
                'f1': float(report[c]['f1-score']),
                'auc': float(roc_auc_score(labels == i, probs[:, i])),
                'support': int(report[c]['support']),
            } for i, c in enumerate(CLASSES)
        },
        'confusion_matrix': confusion_matrix(labels, preds, labels=range(len(CLASSES))).tolist()
    }


def format_metrics(m):
    # format metrics into lines
    lines = [
        f"n={m['n']}  accuracy={m['accuracy']:.4f}  balanced_accuracy={m['balanced_accuracy']:.4f}"
        f"  macro_f1={m['macro_f1']:.4f}  loss={m['loss']:.4f}",
        f"{'class':>8} {'precision':>9} {'recall':>7} {'f1':>7} {'auc':>7} {'support':>8}"
    ]

    # format per-class metrics into lines
    for c, s in m['per_class'].items():
        lines.append(f"{c:>8} {s['precision']:9.4f} {s['recall']:7.4f} {s['f1']:7.4f} {s['auc']:7.4f} {s['support']:8d}")
    
    return "\n".join(lines)


def plot_confusion_matrix(cm, ax=None):
    """
    row-normalised confusion matrix where rows show how each class was predicted
    """
    cm = np.asarray(cm)
    rates = cm / cm.sum(axis=1, keepdims=True)
    labels = [CLASS_LABELS[c] for c in CLASSES]
    ax = ax or plt.subplots(figsize=(5.2, 4.4))[1]

    ax.imshow(rates, cmap=P.BLUES, vmin=0, vmax=1)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 2-pixel gaps between cells
    ax.set_xticks(np.arange(-0.5, len(CLASSES)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASSES)), minor=True)
    ax.grid(which='minor', color=P.SURFACE, linewidth=2)
    ax.tick_params(which='both', length=0)

    # each class with every other class
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = '#ffffff' if rates[i, j] > 0.55 else P.INK
            ax.text(
                j, i, 
                f'{rates[i, j]:.1%}\n', 
                ha='center', va='center',
                color=color, fontsize=11, fontweight='bold'
            )
            ax.text(
                j, i, 
                f'\n{cm[i, j]:,} img', 
                ha='center', va='center',
                color=color, fontsize=8.5
            )

    ax.set_xticks(range(len(CLASSES)), labels)
    ax.set_yticks(range(len(CLASSES)), labels)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('Confusion matrix (test set)', loc='left')
    return ax


def plot_roc_curves(probs, labels, ax=None):
    """
    one-vs-rest ROC curve per class
    """
    ax = ax or plt.subplots(figsize=(5.2, 4.4))[1]
    ax.plot(
        [0, 1], [0, 1], 
        color=P.MUTED, 
        linewidth=1, 
        linestyle=(0, (4, 4)), 
        label="Chance"
    )

    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
        auc = roc_auc_score(labels == i, probs[:, i])
        ax.plot(fpr, tpr, color=P.SERIES[i], label=f'{CLASS_LABELS[c]}  (AUC {auc:.3f}')

    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect('equal')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curves, one-vs-rest (test set)', loc='left')
    ax.legend(loc='lower right', labelcolor=P.INK_SECONDARY)
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.checkpoint, device)

    dataset = ChestXRayDataset(args.split)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    probs, labels = predict(model, loader, device, desc=f"evaluate {args.split}")
    metrics = compute_metrics(probs, labels)

    print(format_metrics(metrics))

    # save results
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f'metrics_{args.split}.json').write_text(json.dumps(metrics, indent=2))
    np.savez(
        RESULTS_DIR / f'predictions_{args.split}.npz', 
        probs=probs, 
        labels=labels,
        paths=np.array(dataset.paths)
    )

    # plot and save confusion matrix and roc curves
    if args.split == 'test':
        ASSETS_DIR.mkdir(exist_ok=True)
        plot_confusion_matrix(metrics['confusion_matrix'])
        plt.savefig(ASSETS_DIR / 'confusion_matrix.png')
        plot_roc_curves(probs, labels)
        plt.savefig(ASSETS_DIR / 'roc_curves.png')
        print(f'Figures saved to {ASSETS_DIR}')


if __name__ == '__main__':
    main()
