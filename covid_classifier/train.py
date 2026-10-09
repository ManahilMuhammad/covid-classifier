"""
finetune resnet18 and keep checkpoint with best validation score
"""

import argparse
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from . import plotting as P
from .config import ASSETS_DIR, CHECKPOINT, RESULTS_DIR
from .data import make_loaders
from .evaluate import compute_metrics, predict
from .model import build_model, get_device, save_checkpoint


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, loss_fn, optimiser, device, desc="train"):
    model.train()

    total_loss, correct, seen = 0.0, 0, 0
    progress = tqdm(loader, desc=desc, leave=False)

    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        seen += len(labels)
        progress.set_postfix(loss=f"{total_loss / seen:.4f}")
    return total_loss / seen, correct / seen


def plot_history(history, path=None):
    """
    loss and accuracy per epoch as side by side panels 
    """
    epochs = [h['epoch'] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(9, 3.4))

    for ax, key, title in [(ax_loss, 'loss', 'Loss'), (ax_acc, 'accuracy', 'Accuracy')]:

        ax.plot(
            epochs, [h[f'train_{key}'] for h in history],
            color=P.SERIES[0], marker='o', markersize=5, 
            label='Train'
        )

        ax.plot(epochs, [h[f'val_{key}'] for h in history], 
                color=P.SERIES[1], marker='o', markersize=5, 
                label='Validation'
        )

        ax.set_title(title, loc='left')
        ax.set_xlabel('Epoch')
        ax.set_xticks(epochs)

    ax_acc.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax_loss.legend(labelcolor=P.INK_SECONDARY)
    fig.tight_layout()

    if path:
        fig.savefig(path)

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--samples-per-epoch', type=int, default=3000,
                        help='training images drawn (class-balanced) per epoch')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed) # reproducible sampling
    device = get_device()
    print(f'Device: {device}')

    # test loader not used because only used for evaluation
    dl_train, dl_val, _ = make_loaders(args.batch_size, args.num_workers, args.samples_per_epoch)
    model = build_model().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    history, best = [], -1.0 # -1 so first epoch always saved
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, dl_train, loss_fn, optimiser, device,
            desc=f'epoch {epoch}/{args.epochs}'
        )

        scheduler.step()
        val = compute_metrics(*predict(model, dl_val, device, desc="validate"))

        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc,
                        'val_loss': val['loss'], 'val_accuracy': val['accuracy'],
                        'val_balanced_accuracy': val['balanced_accuracy'],
                        'val_macro_f1': val['macro_f1'], 'seconds': time.time() - start})

        # model selection on validation balanced accuracy
        improved = val['balanced_accuracy'] > best
        print(f'Epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.4f} | '
              f"val loss {val['loss']:.4f} acc {val['accuracy']:.4f} "
              f"balanced acc {val['balanced_accuracy']:.4f} | {time.time() - start:.0f}s"
              + ('  * saved' if improved else ''))
        
        if improved:
            best = val['balanced_accuracy']
            save_checkpoint(model, CHECKPOINT, epoch=epoch, val_metrics=val, args=vars(args))

    # save results
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / 'history.json').write_text(json.dumps(history, indent=2))
    ASSETS_DIR.mkdir(exist_ok=True)
    plot_history(history, ASSETS_DIR / 'training_curves.png')
    print(f'Best val balanced accuracy {best:.4f}; checkpoint at {CHECKPOINT}')


if __name__ == '__main__':
    main()
