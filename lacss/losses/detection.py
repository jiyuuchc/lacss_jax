from functools import partial
from typing import Optional

import jax
import optax

from ..train.loss import Loss

jnp = jax.numpy

EPS = jnp.finfo("float32").eps


def _binary_focal_crossentropy(pred, gt, gamma=2.0):
    p_t = gt * pred + (1 - gt) * (1.0 - pred)
    focal_factor = (1.0 - p_t) ** gamma

    bce = -jnp.log(jnp.clip(p_t, EPS, 1.0))

    return focal_factor * bce


def detection_loss(preds, gamma=2.0, **kwargs):
    """LPN detection loss"""

    scores = preds["lpn_scores"]
    gt_scores = preds["lpn_gt_scores"]

    score_loss = 0.0
    cnt = EPS
    for k in scores:
        score_loss += _binary_focal_crossentropy(
            scores[k],
            gt_scores[k],
            gamma,
        ).sum()
        cnt += preds["lpn_scores"][k].size

    return score_loss / cnt


def localization_loss(preds, delta=1.0, **kwargs):
    """LPN localization loss"""

    regrs = preds["lpn_regressions"]
    gt_regrs = preds["lpn_gt_regressions"]
    gt_scores = preds["lpn_gt_scores"]

    regr_loss = 0.0
    cnt = 1e-8
    for k in regrs:
        # h = optax.l2_loss(regrs[k], gt_regrs[k])
        h = optax.huber_loss(regrs[k], gt_regrs[k], delta=delta).mean(
            axis=-1, keepdims=True
        )
        mask = gt_scores[k] > 0
        regr_loss += jnp.sum(h, where=mask)
        cnt += jnp.count_nonzero(mask)

    return regr_loss / (cnt + EPS)


def lpn_loss(preds, gamma=2.0, w1=1.0, w2=1.0, **kwargs):
    """LPN loss"""

    return detection_loss(preds, gamma=gamma) * w1 + localization_loss(preds) * w2


class DetectionLoss(Loss):
    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, preds: dict, **kwargs) -> jnp.ndarray:

        return detection_loss(preds=preds, gamma=self.gamma)


class LocalizationLoss(Loss):
    def __init__(self, delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def call(self, preds: dict, **kwrags) -> jnp.ndarray:

        return localization_loss(preds, delta=self.delta)


class LPNLoss(Loss):
    def __init__(self, delta=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.det_loss = DetectionLoss(gamma=gamma)
        self.loc_loss = LocalizationLoss(delta=delta)

    def call(self, preds: dict, **kwargs):
        return self.det_loss.call(preds=preds) + self.loc_loss(preds=preds)
