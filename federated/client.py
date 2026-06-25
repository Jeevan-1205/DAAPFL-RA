"""Flower NumPyClient for DAAPFL-RA and baselines.

A client owns one disaster partition. It keeps PRIVATE params (decoder + head
+ prototype state) locally across rounds and only exchanges SHARED params
(encoder) with the server. For DAAPFL-RA it additionally returns its
multi-prototype matrix and validation damage-F1 in the fit metrics so the
server can compute attention + reliability weights.
"""
from __future__ import annotations
import base64, pickle
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import flwr as fl
from flwr.common import NDArrays, Scalar

from models import build_model
from models.prototype import MultiPrototype, compute_client_prototypes
from losses import build_loss
from training.trainer import build_optimizer, local_train
from evaluation.metrics import evaluate_model
from utils.param_utils import (state_dict_to_ndarrays, ndarrays_to_state_dict,
                               filter_state_dict)


def _encode(obj) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def _decode(s: str):
    return pickle.loads(base64.b64decode(s.encode("ascii")))


class DamageClient(fl.client.NumPyClient):
    def __init__(self, cid: int, train_loader, val_loader, cfg,
                 device: str = "cuda", method: str = "daapfl_ra",
                 private_patterns: Optional[List[str]] = None):
        self.cid = cid
        self.cfg = cfg
        self.device = device if torch.cuda.is_available() else "cpu"
        self.method = method
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = build_model(cfg).to(self.device)
        self.loss_fn = build_loss(cfg).to(self.device)
        self.private_patterns = private_patterns or list(
            cfg.model.private_param_patterns)
        # shared param keys = encoder-only (everything NOT private)
        self.shared_keys = list(filter_state_dict(
            self.model.state_dict(), self.private_patterns, keep=False).keys())
        # prototype state for DAAPFL-RA
        k = int(cfg.get_path("daapfl_ra.num_prototypes", 3))
        dim = self.model.bottleneck_dim
        mom = float(cfg.get_path("daapfl_ra.prototype_momentum", 0.9))
        self.proto = MultiPrototype(k, dim, mom)
        self.num_train = sum(len(b["disaster"]) for b in []) or self._count(train_loader)

    @staticmethod
    def _count(loader) -> int:
        try:
            return len(loader.dataset)
        except Exception:
            return 1

    # ---------- shared param exchange ----------
    def get_parameters(self, config: Dict = None) -> NDArrays:
        shared = filter_state_dict(self.model.state_dict(),
                                   self.private_patterns, keep=False)
        return state_dict_to_ndarrays(shared)

    def _set_shared(self, parameters: NDArrays) -> None:
        sd = ndarrays_to_state_dict(self.shared_keys, parameters)
        self.model.load_state_dict(sd, strict=False)  # private params untouched

    # ---------- prototype computation ----------
    @torch.no_grad()
    def _compute_prototypes(self, max_batches: int = 8) -> torch.Tensor:
        self.model.eval()
        feats = []
        for i, batch in enumerate(self.train_loader):
            if i >= max_batches:
                break
            pre = batch["pre"].to(self.device); post = batch["post"].to(self.device)
            feats.append(self.model.extract_bottleneck(pre, post).cpu())
        if not feats:
            return self.proto.get()
        f = torch.cat(feats, 0)
        k = self.proto.k
        new = compute_client_prototypes(f, k=k, seed=int(self.cfg.seed))
        return self.proto.update(new)

    # ---------- FL hooks ----------
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]
            ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self._set_shared(parameters)
        opt = build_optimizer(self.model, self.cfg)
        epochs = int(self.cfg.federated.local_epochs)
        amp = bool(self.cfg.federated.amp)
        gc = float(self.cfg.train.get("grad_clip", 1.0))
        prox = self._proximal_term(parameters) if self.method == "fedprox" else None
        stats = local_train(self.model, self.train_loader, self.loss_fn, opt,
                            self.device, epochs=epochs, amp=amp,
                            grad_clip=gc, proximal=prox)
        metrics: Dict[str, Scalar] = {"loss": stats["loss"], "cid": self.cid}
        # validation F1 (reliability) + prototypes (similarity) for DAAPFL-RA
        if self.method == "daapfl_ra":
            f1 = self._val_f1()
            protos = self._compute_prototypes()
            metrics["val_f1_dam"] = f1
            metrics["prototypes"] = _encode(protos.cpu().numpy())
        return self.get_parameters(), stats["num_samples"], metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]
                 ) -> Tuple[float, int, Dict[str, Scalar]]:
        self._set_shared(parameters)
        if self.val_loader is None:
            return 0.0, 1, {}
        m = evaluate_model(self.model, self.val_loader, self.device,
                           int(self.cfg.model.num_classes))
        n = self._count(self.val_loader)
        return float(1.0 - m["overall"]), n, {f"val_{k}": v for k, v in m.items()}

    # ---------- helpers ----------
    def _val_f1(self) -> float:
        if self.val_loader is None:
            return 0.0
        m = evaluate_model(self.model, self.val_loader, self.device,
                           int(self.cfg.model.num_classes))
        return float(m["f1_dam"])

    def _proximal_term(self, global_params: NDArrays):
        mu = float(self.cfg.get_path("fedprox.mu", 0.01))
        global_t = [torch.as_tensor(a).to(self.device) for a in global_params]
        keys = self.shared_keys

        def prox(model):
            sd = model.state_dict()
            reg = torch.zeros((), device=self.device)
            for k, g in zip(keys, global_t):
                reg = reg + ((sd[k] - g) ** 2).sum()
            return 0.5 * mu * reg
        return prox


def make_client_fn(partitions, cfg, method: str, device: str = "cuda"):
    """Returns a Flower client_fn building a DamageClient per cid."""
    from datasets.dataloaders import make_client_loaders
    cache = {}

    def client_fn(cid: str):
        idx = int(cid)
        if idx not in cache:
            tr, va = make_client_loaders(partitions[idx], cfg)
            cache[idx] = (tr, va)
        tr, va = cache[idx]
        return DamageClient(idx, tr, va, cfg, device=device, method=method).to_client()
    return client_fn
