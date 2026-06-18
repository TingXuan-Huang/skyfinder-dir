"""Camera-conditioned model: image features + a learnable per-camera embedding.

A per-camera embedding is projected and ADDED to the backbone feature before the head.
LOCO test cameras (unseen in train) get an 'unknown' token (index 0), trained via random
cam-dropout so it sees real gradients. Tests whether a per-camera prior closes the LOCO gap.

Training driver: cam_cond_train.py.  Smoke: python -m skyfinder.training.cam_conditioned --smoke
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .model import build_model


class CamConditionedModel(nn.Module):
    """Backbone image feature `f_img` + projected camera embedding `W e_cam`, read by the head.

    cam_dropout_prob>0 randomly replaces train cameras with the unknown token (idx 0) so the
    unknown embedding — used for all LOCO test cameras — gets real gradients.
    """

    def __init__(self, backbone_name, cam_id_to_idx, emb_dim=64,
                 cam_dropout_prob=0.05, freeze_backbone=False):
        super().__init__()
        base = build_model(backbone_name, freeze_backbone=freeze_backbone)
        if hasattr(base, "fc") and isinstance(base.fc, nn.Linear):
            self.feat_dim = base.fc.in_features
            base.fc = nn.Identity()
        elif hasattr(base, "heads"):
            self.feat_dim = base.heads.head.in_features
            base.heads.head = nn.Identity()
        else:
            raise ValueError(f"unsupported backbone: {backbone_name}")
        self.backbone = base

        self.cam_id_to_idx = dict(cam_id_to_idx)  # raw CamId -> idx>=1; idx 0 = unknown
        self.n_cams = max(cam_id_to_idx.values()) + 1
        self.unknown_idx = 0
        assert self.unknown_idx not in cam_id_to_idx.values(), "reserve idx 0 for unknown"
        self.cam_emb = nn.Embedding(self.n_cams, emb_dim)
        self.cam_proj = nn.Linear(emb_dim, self.feat_dim)
        self.head = nn.Linear(self.feat_dim, 1)
        self.cam_dropout_prob = cam_dropout_prob

    def cam_idx(self, cam_ids: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(cam_ids, dtype=torch.long)
        for i, c in enumerate(cam_ids.tolist()):
            out[i] = self.cam_id_to_idx.get(int(c), self.unknown_idx)
        return out

    def forward(self, x: torch.Tensor, cam_ids: torch.Tensor) -> torch.Tensor:
        f_img = self.backbone(x)
        if f_img.ndim > 2:
            f_img = f_img.flatten(1)
        idx = self.cam_idx(cam_ids).to(x.device)
        if self.training and self.cam_dropout_prob > 0:
            mask = torch.rand_like(idx, dtype=torch.float32) < self.cam_dropout_prob
            idx = torch.where(mask, torch.full_like(idx, self.unknown_idx), idx)
        e = self.cam_proj(self.cam_emb(idx))
        return self.head(f_img + e).squeeze(-1)


def build_cam_id_to_idx(train_cam_ids) -> dict:
    """Each train camera -> idx>=1; idx 0 reserved for the unknown token."""
    return {int(c): i + 1 for i, c in enumerate(sorted(set(int(c) for c in train_cam_ids)))}


def _smoke():
    idx = build_cam_id_to_idx([10, 11, 12, 13, 14, 15])
    m = CamConditionedModel("resnet50", idx, emb_dim=8, cam_dropout_prob=0.5)
    cam_t = torch.tensor([10, 11, 12, 13, 14, 15, 99], dtype=torch.long)  # 99 unseen
    x = torch.randn(len(cam_t), 3, 224, 224)
    m.train()
    out = m(x, cam_t)
    print("train out:", tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))
    m.eval()
    with torch.no_grad():
        a = m(x[-1:], cam_t[-1:]).item()
        b = m(x[-1:], cam_t[-1:]).item()
    print("unseen-cam pred deterministic at eval:", abs(a - b) < 1e-5)
    print("smoke OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    if ap.parse_args().smoke:
        _smoke()
