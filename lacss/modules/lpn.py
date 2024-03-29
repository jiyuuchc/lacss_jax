import typing as tp
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..ops import locations_to_labels
from .common import ChannelAttention

# from typing import List, Optional, Sequence, Tuple, Union


class LPN(nn.Module):
    """
    Args:
    in_channels: number of channels in input feature
    feature_level: input feature level default 3
    conv_layers: conv layer spec
    with_channel_attention: whether include channel attention
    """

    feature_levels: tp.Sequence[int] = (4, 3, 2)
    conv_spec: tp.Tuple[tp.Sequence[int], tp.Sequence[int]] = ((384, 384, 384, 384), ())
    detection_roi: float = 8.0

    @nn.compact
    def _process_feature(
        self, feature: jnp.ndarray
    ) -> tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        conv_spec = self.conv_spec

        x = feature

        for n_ch in conv_spec[0]:
            x = nn.Conv(n_ch, (3, 3), use_bias=False)(x)
            x = nn.GroupNorm(num_groups=n_ch)(x[None, ...])[0]
            x = jax.nn.relu(x)

        for n_ch in conv_spec[1]:
            x = nn.Conv(n_ch, (1, 1), use_bias=False)(x)
            x = nn.GroupNorm(num_groups=n_ch)(x[None, ...])[0]
            x = jax.nn.relu(x)

        scores_out = nn.Conv(1, (1, 1))(x)
        scores_out = jax.nn.sigmoid(scores_out)

        regression_out = nn.Conv(2, (1, 1))(x)

        return scores_out, regression_out, x

    def __call__(
        self,
        inputs: dict,
        scaled_gt_locations: jnp.ndarray = None,
    ) -> dict:
        """
        Args:
            inputs: feature dict: {'lvl': [H, W, C]}
            scaled_gt_locations: scaled 0..1 [N, 2], only valid in training
        Returns:
            outputs:
            {
                lpn_scores: dict: {'lvl': [H, W, 1]}
                lpn_regressions: dict {'lvl': [H, W, 2]}
                gt_lpn_scores: dict {'lvl': [H, W, 1]}, only if training
                gt_lpn_regressions: dict {'lvl': [H, W, 2]}, only if training
            }
        """

        all_scores = dict()
        all_regrs = dict()
        all_features = dict()

        for lvl in self.feature_levels:

            feature = inputs[str(lvl)]
            score, regression, lpn_feature = self._process_feature(feature)
            all_scores[str(lvl)] = score
            all_regrs[str(lvl)] = regression
            all_features[str(lvl)] = lpn_feature

        outputs = dict(
            lpn_features=all_features,
            lpn_scores=all_scores,
            lpn_regressions=all_regrs,
        )

        if scaled_gt_locations is not None:

            all_gt_scores = dict()
            all_gt_regrs = dict()

            for lvl in self.feature_levels:

                feature_shape = inputs[str(lvl)].shape[-3:-1]

                score_target, regression_target = locations_to_labels(
                    scaled_gt_locations,
                    target_shape=feature_shape,
                    threshold=self.detection_roi / (2**lvl),
                )
                all_gt_scores[str(lvl)] = score_target
                all_gt_regrs[str(lvl)] = regression_target

            outputs.update(
                dict(
                    lpn_gt_scores=all_gt_scores,
                    lpn_gt_regressions=all_gt_regrs,
                )
            )

        return outputs
