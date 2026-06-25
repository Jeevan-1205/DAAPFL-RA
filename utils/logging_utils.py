"""Lightweight console logger + optional wandb wrapper."""
from __future__ import annotations
import logging, sys
from typing import Any, Dict, Optional

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "daapfl", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class WandbLogger:
    """No-op when disabled or wandb missing; otherwise thin passthrough."""
    def __init__(self, enabled: bool, project: str = "daapfl-ra",
                 run_name: Optional[str] = None, config: Optional[Dict] = None):
        self.enabled = enabled
        self._run = None
        if not enabled:
            return
        try:
            import wandb
            self._wandb = wandb
            self._run = wandb.init(project=project, name=run_name, config=config or {})
        except Exception as e:  # pragma: no cover
            get_logger().warning("wandb disabled (%s)", e)
            self.enabled = False

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self._run is not None:
            self._wandb.log(data, step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._wandb.finish()
