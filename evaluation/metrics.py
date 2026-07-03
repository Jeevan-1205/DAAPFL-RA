"""xBD-style metrics.

F1-loc : binary building-vs-background F1.
F1-dam : macro F1 over the 4 damage classes (computed on building pixels).
overall: xView2 score = 0.3*F1loc + 0.7*F1dam (configurable).
fairness: Jain's fairness index over per-client (or per-class) scores.

Extended metrics (v2):
  precision / recall per class
  IoU per class + mIoU
  confusion matrix
  optional validation loss in evaluate_model()
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np
import torch


@dataclass
class SegMetrics:
    num_classes: int = 5
    eps: float = 1e-7
    # confusion accumulators
    tp: np.ndarray = field(default=None)
    fp: np.ndarray = field(default=None)
    fn: np.ndarray = field(default=None)

    def __post_init__(self):
        c = self.num_classes
        self.tp = np.zeros(c); self.fp = np.zeros(c); self.fn = np.zeros(c)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        pred = logits.argmax(1)
        p = pred.flatten().cpu().numpy()
        t = target.flatten().cpu().numpy()
        for c in range(self.num_classes):
            pc, tc = (p == c), (t == c)
            self.tp[c] += np.sum(pc & tc)
            self.fp[c] += np.sum(pc & ~tc)
            self.fn[c] += np.sum(~pc & tc)

    # ---- per-class metrics ---- #

    def _f1(self, c: int) -> float:
        tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
        prec = tp / (tp + fp + self.eps)
        rec = tp / (tp + fn + self.eps)
        return float(2 * prec * rec / (prec + rec + self.eps))

    def _precision(self, c: int) -> float:
        tp, fp = self.tp[c], self.fp[c]
        return float(tp / (tp + fp + self.eps))

    def _recall(self, c: int) -> float:
        tp, fn = self.tp[c], self.fn[c]
        return float(tp / (tp + fn + self.eps))

    def _iou(self, c: int) -> float:
        tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
        return float(tp / (tp + fp + fn + self.eps))

    def per_class_f1(self) -> List[float]:
        return [self._f1(c) for c in range(self.num_classes)]

    def per_class_precision(self) -> List[float]:
        return [self._precision(c) for c in range(self.num_classes)]

    def per_class_recall(self) -> List[float]:
        return [self._recall(c) for c in range(self.num_classes)]

    def per_class_iou(self) -> List[float]:
        return [self._iou(c) for c in range(self.num_classes)]

    # ---- aggregate metrics ---- #

    def f1_localization(self) -> float:
        # building = any class >=1 ; aggregate damage classes vs background
        tp = self.tp[1:].sum(); fp = self.fp[1:].sum(); fn = self.fn[1:].sum()
        prec = tp / (tp + fp + self.eps); rec = tp / (tp + fn + self.eps)
        return float(2 * prec * rec / (prec + rec + self.eps))

    def f1_damage(self) -> float:
        return float(np.mean([self._f1(c) for c in range(1, self.num_classes)]))

    def miou(self) -> float:
        """Mean Intersection-over-Union across all classes."""
        return float(np.mean(self.per_class_iou()))

    def overall(self, w_loc: float = 0.3, w_dam: float = 0.7) -> float:
        return w_loc * self.f1_localization() + w_dam * self.f1_damage()

    def confusion_matrix(self) -> np.ndarray:
        """Reconstruct an approximate confusion matrix from tp/fp/fn.

        For a full pixel-level confusion matrix, use
        ``build_confusion_matrix()`` below which accumulates during
        evaluation.  This method returns per-class (tp, fp, fn) as a
        (num_classes, 3) array for lightweight inspection.
        """
        return np.stack([self.tp, self.fp, self.fn], axis=1)

    def summary(self) -> Dict[str, float]:
        pcf = self.per_class_f1()
        pcp = self.per_class_precision()
        pcr = self.per_class_recall()
        pci = self.per_class_iou()
        return {
            "f1_loc": self.f1_localization(),
            "f1_dam": self.f1_damage(),
            "overall": self.overall(),
            "miou": self.miou(),
            **{f"f1_class_{c}": pcf[c] for c in range(self.num_classes)},
            **{f"precision_class_{c}": pcp[c] for c in range(self.num_classes)},
            **{f"recall_class_{c}": pcr[c] for c in range(self.num_classes)},
            **{f"iou_class_{c}": pci[c] for c in range(self.num_classes)},
        }


def f1_localization(metrics: SegMetrics) -> float:
    return metrics.f1_localization()


def f1_damage(metrics: SegMetrics) -> float:
    return metrics.f1_damage()


def per_class_f1(metrics: SegMetrics) -> List[float]:
    return metrics.per_class_f1()


def fairness_index(scores: Sequence[float], eps: float = 1e-12) -> float:
    """Jain's fairness index in [1/n, 1]; 1 == perfectly equal."""
    x = np.asarray(scores, dtype=np.float64)
    if x.size == 0:
        return 0.0
    num = (x.sum()) ** 2
    den = x.size * (x ** 2).sum() + eps
    return float(num / den)


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    num_classes: int = 5,
    loss_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """Evaluate a model on a data loader.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : str
    num_classes : int
    loss_fn : callable, optional
        If provided, validation loss is computed and included in the
        returned dict under the key ``"loss"``.  When ``None`` (the
        default), ``"loss"`` is omitted for backward compatibility.

    Returns
    -------
    Dict[str, float]
        Metric dictionary from ``SegMetrics.summary()``, optionally
        augmented with ``"loss"``.
    """
    model.eval()
    m = SegMetrics(num_classes=num_classes)
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(pre, post)
        m.update(logits, mask)
        if loss_fn is not None:
            total_loss += float(loss_fn(logits, mask).item())
            n_batches += 1
    result = m.summary()
    if loss_fn is not None:
        result["loss"] = total_loss / max(n_batches, 1)
    return result
