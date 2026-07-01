"""
Base client for pure-PyTorch federated learning.

Wraps a single disaster partition with its own model, optimizer,
data loaders, and prototype state. Communicates with the server
exclusively through ClientUpdate dataclasses.

Reuses
------
- training.trainer.local_train      (local SGD loop)
- training.trainer.build_optimizer  (config-driven optimizer)
- evaluation.metrics.evaluate_model (F1-loc / F1-dam / overall)
- models.build_model                (SiameseUNet construction)
- models.prototype                  (multi-prototype KMeans + EMA)
- losses.build_loss                 (hybrid / focal / dice)
- datasets.dataloaders.make_client_loaders
- utils.param_utils.filter_state_dict
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from datasets.dataloaders import make_client_loaders
from datasets.partition import ClientPartition
from evaluation.metrics import evaluate_model
from losses import build_loss
from models import build_model
from models.prototype import MultiPrototype, compute_client_prototypes
from training.trainer import build_optimizer, local_train
from utils.param_utils import filter_state_dict

from federated.update import ClientUpdate


class FedClient:
    """
    A single federated client (one disaster partition).

    Lifecycle per round
    -------------------
    1. Server calls ``load_encoder(state_dict)`` to push global /
       personalized encoder weights.
    2. Server calls ``fit()`` which runs ``local_train()`` and returns
       a ``ClientUpdate``.
    3. Server calls ``evaluate()`` for validation metrics.

    Private parameters (decoder, segmentation head, prototype state)
    are never exchanged — they persist on this object across rounds.
    """

    def __init__(
        self,
        client_id: int,
        partition: ClientPartition,
        cfg,
        device: str = "cuda",
        method: str = "fedavg",
    ):
        self.client_id = client_id
        self.cfg = cfg
        self.device = device if torch.cuda.is_available() else "cpu"
        self.method = method

        # ---- model + loss (private state lives here across rounds) ----
        self.model = build_model(cfg).to(self.device)
        self.loss_fn = build_loss(cfg).to(self.device)

        # ---- data ----
        self.train_loader, self.val_loader = make_client_loaders(
            partition, cfg,
        )
        self.num_samples = len(self.train_loader.dataset)

        # ---- shared / private key sets ----
        self.private_patterns: List[str] = list(
            cfg.model.private_param_patterns,
        )

        # ---- prototype state (DAAPFL-RA) ----
        k = int(cfg.get_path("daapfl_ra.num_prototypes", 3))
        dim = self.model.bottleneck_dim
        mom = float(cfg.get_path("daapfl_ra.prototype_momentum", 0.9))
        self.proto = MultiPrototype(k, dim, mom)

        # ---- pre-training encoder snapshot (for proximal term) ----
        self._global_encoder: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------ #
    # Encoder exchange                                                     #
    # ------------------------------------------------------------------ #

    def get_encoder_state(self) -> Dict[str, torch.Tensor]:
        """
        Return shared (encoder-only) parameters as an OrderedDict.
        """

        return filter_state_dict(
            self.model.state_dict(),
            self.private_patterns,
            keep=False,
        )

    def load_encoder(self, state: Dict[str, torch.Tensor]) -> None:
        """
        Load aggregated encoder weights.

        Private params (decoder / head / prototype) are untouched.
        Also snapshots the encoder for proximal regularization.
        """

        self.model.load_state_dict(state, strict=False)
        self._global_encoder = {
            k: v.clone().detach() for k, v in state.items()
        }

    # ------------------------------------------------------------------ #
    # Local training                                                       #
    # ------------------------------------------------------------------ #

    def fit(self) -> ClientUpdate:
        """
        Run local training and return a ClientUpdate.

        Delegates to ``training.trainer.local_train``.
        Populates val_metrics and prototypes for algorithms
        that need them (e.g. DAAPFL-RA).
        """

        optimizer = build_optimizer(self.model, self.cfg)

        epochs = int(self.cfg.federated.local_epochs)
        amp = bool(self.cfg.federated.amp)
        grad_clip = float(self.cfg.train.get("grad_clip", 1.0))

        # -- proximal term (FedProx / Ditto / pFedMe) --
        proximal = self._build_proximal()

        stats = local_train(
            self.model,
            self.train_loader,
            self.loss_fn,
            optimizer,
            self.device,
            epochs=epochs,
            amp=amp,
            grad_clip=grad_clip,
            proximal=proximal,
        )

        # -- validation metrics --
        val_metrics = self.evaluate()

        # -- prototypes --
        prototypes = self.compute_prototypes()

        # -- build update --
        update = ClientUpdate(
            client_id=self.client_id,
            encoder_state=self.get_encoder_state(),
            num_samples=stats["num_samples"],
            train_loss=stats["loss"],
            val_metrics=val_metrics,
            prototypes=prototypes,
        )

        return update

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on the local validation set.

        Returns the metric dict from ``evaluation.metrics.evaluate_model``
        (keys: f1_loc, f1_dam, overall, f1_class_0 … f1_class_4).
        Returns an empty dict if no validation data exists.
        """

        if self.val_loader is None:
            return {}

        return evaluate_model(
            self.model,
            self.val_loader,
            self.device,
            int(self.cfg.model.num_classes),
        )

    # ------------------------------------------------------------------ #
    # Prototype computation (DAAPFL-RA)                                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def compute_prototypes(self, max_batches: int = 8) -> torch.Tensor:
        """
        Compute K multi-prototypes from encoder bottleneck features.

        Returns
        -------
        torch.Tensor
            Shape ``(K, D)`` — the EMA-updated prototype matrix.
        """

        self.model.eval()

        features: List[torch.Tensor] = []

        for i, batch in enumerate(self.train_loader):
            if i >= max_batches:
                break

            pre = batch["pre"].to(self.device)
            post = batch["post"].to(self.device)

            features.append(
                self.model.extract_bottleneck(pre, post).cpu(),
            )

        if not features:
            return self.proto.get()

        f = torch.cat(features, dim=0)
        k = self.proto.k

        new = compute_client_prototypes(
            f, k=k, seed=int(self.cfg.seed),
        )

        return self.proto.update(new)

    # ------------------------------------------------------------------ #
    # Proximal regularization (FedProx / Ditto / pFedMe)                   #
    # ------------------------------------------------------------------ #

    def _build_proximal(self):
        """
        Build a proximal regularizer closure for ``local_train()``.

        Returns ``None`` for methods that don't need it. For FedProx
        and similar, returns a callable ``proximal(model) -> Tensor``
        that computes ``mu/2 * ||w - w_global||^2``.
        """

        if self.method not in ("fedprox", "ditto", "pfedme"):
            return None

        if self._global_encoder is None:
            return None

        mu = float(self.cfg.get_path("fedprox.mu", 0.01))
        device = self.device
        global_tensors = {
            k: v.to(device) for k, v in self._global_encoder.items()
        }

        def proximal(model):
            sd = model.state_dict()
            reg = torch.zeros((), device=device)
            for k, g in global_tensors.items():
                if k in sd:
                    reg = reg + ((sd[k] - g) ** 2).sum()
            return 0.5 * mu * reg

        return proximal
