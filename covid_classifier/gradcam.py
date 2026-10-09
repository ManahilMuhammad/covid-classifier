"""
highlight image regions that drove prediction

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2020). 
Grad-CAM: Visual explanations from deep networks via gradient-based localization. 
International Journal of Computer Vision, 128(2), 336-359. 
https://doi.org/10.1007/s11263-019-01228-7
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from . import plotting as P
from .config import ASSETS_DIR, CHECKPOINT, CLASS_LABELS, CLASSES, RESULTS_DIR
from .data import ChestXRayDataset, denormalize
from .model import get_device, load_model

HEAT_COLOR = np.array([0.92, 0.41, 0.20])  # single hue; intensity is carried by opacity


class GradCAM:
    """
    usage:
        with GradCAM(model) as cam:
            heatmaps, probs = cam(images)
    """

    def __init__(self, model, layer=None):
        self.model = model
        self._handle = (layer or model.layer4).register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(lambda grad: setattr(self, "gradients", grad))

    def __call__(self, images, class_idx=None):
        """
        return (heatmaps [N, H, W] in [0, 1], probabilities [N, C])
        - explains class_idx if given otherwise each image's predicted class
        """
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(images)
            if class_idx is None:
                class_idx = logits.argmax(1)
            class_idx = torch.as_tensor(class_idx, device=logits.device).view(-1, 1)
            logits.gather(1, class_idx.expand(len(logits), 1)).sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True) # weights are spatially averaged gradients

        # cam is ReLU(weighted sum of activations)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)

        return cam.detach().cpu().numpy(), torch.softmax(logits, 1).detach().cpu().numpy()

    def remove(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def overlay(image, heatmap, alpha=0.65):
    """
    blend a heatmap onto an hwc image
    """
    a = (alpha * heatmap)[..., None]
    return image * (1 - a) + HEAT_COLOR * a


def pick_examples(probs, labels, n_per_class=2, n_errors=2, seed=0):
    """
    indices of a few correct predictions per class plus some misclassifications
    """
    rng = np.random.default_rng(seed)
    preds = probs.argmax(1)
    picked = []

    for i in range(len(CLASSES)):
        correct = np.flatnonzero((labels == i) & (preds == i))
        picked += rng.choice(correct, min(n_per_class, len(correct)), replace=False).tolist()

    wrong = np.flatnonzero(preds != labels)
    picked += rng.choice(wrong, min(n_errors, len(wrong)), replace=False).tolist()

    return picked


def plot_gradcam(images, heatmaps, probs, labels):
    """
    two rows per example: x-ray then the gradcam overlay
    """
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.9))
    axes = np.asarray(axes).reshape(2, n)

    for k in range(n):
        img = denormalize(images[k])
        pred = int(probs[k].argmax())
        true = int(labels[k])
        mark = "" if pred == true else '  X'

        axes[0, k].imshow(img)
        axes[1, k].imshow(overlay(img, heatmaps[k]))
        axes[0, k].set_title(
            f'True: {CLASS_LABELS[CLASSES[true]]}\n'
            f'Pred: {CLASS_LABELS[CLASSES[pred]]} {probs[k, pred]:.0%}{mark}',
            fontsize=9, fontweight='normal', loc='center',
            color=P.INK if pred == true else '#d03b3b'
        )
        
        for ax in axes[:, k]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

            for spine in ax.spines.values():
                spine.set_visible(False)

    axes[0, 0].set_ylabel("X-ray", fontsize=10)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=10)
    fig.tight_layout(h_pad=0.4, w_pad=0.4)

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--n-per-class", type=int, default=2)
    parser.add_argument("--n-errors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # save results
    preds_file = RESULTS_DIR / 'predictions_test.npz'
    if not preds_file.exists():
        raise SystemExit('Run python -m covid_classifier.evaluate first')
    saved = np.load(preds_file)

    device = get_device()
    model = load_model(args.checkpoint, device)

    dataset = ChestXRayDataset("test")
    idx = pick_examples(saved["probs"], saved["labels"], args.n_per_class, args.n_errors, args.seed)
    
    images = torch.stack([dataset[i][0] for i in idx]).to(device)
    labels = saved["labels"][idx]

    with GradCAM(model) as cam:
        heatmaps, probs = cam(images)

    # plot and save gradcams
    plot_gradcam(images, heatmaps, probs, labels)
    ASSETS_DIR.mkdir(exist_ok=True)
    plt.savefig(ASSETS_DIR / 'gradcam.png')
    print(f"Saved {ASSETS_DIR / 'gradcam.png'}")


if __name__ == '__main__':
    main()
