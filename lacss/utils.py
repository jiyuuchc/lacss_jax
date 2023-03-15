import re
import typing as tp

import numpy as np

from .ops import bboxes_of_patches

InputLike = tp.Union[tp.Any, tp.Tuple[tp.Any, ...], tp.Dict[str, tp.Any], "Inputs"]


def _unique_name(
    names: tp.Set[str],
    name: str,
):

    if name in names:

        match = re.match(r"(.*?)(\d*)$", name)
        assert match is not None

        name = match[1]
        num_part = match[2]

        i = int(num_part) if num_part else 2
        str_template = f"{{name}}{{i:0{len(num_part)}}}"

        while str_template.format(name=name, i=i) in names:
            i += 1

        name = str_template.format(name=name, i=i)

    names.add(name)
    return name


def _unique_names(
    names: tp.Iterable[str],
    *,
    existing_names: tp.Optional[tp.Set[str]] = None,
) -> tp.Iterable[str]:
    if existing_names is None:
        existing_names = set()

    for name in names:
        yield _unique_name(existing_names, name)


def _lower_snake_case(s: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    parts = s.split("_")
    output_parts = []

    for i in range(len(parts)):
        if i == 0 or len(parts[i - 1]) > 1:
            output_parts.append(parts[i])
        else:
            output_parts[-1] += parts[i]

    return "_".join(output_parts)


def _get_name(obj) -> str:
    if hasattr(obj, "name") and obj.name:
        return obj.name
    elif hasattr(obj, "__name__") and obj.__name__:
        return _lower_snake_case(obj.__name__)
    elif hasattr(obj, "__class__") and obj.__class__.__name__:
        return _lower_snake_case(obj.__class__.__name__)
    else:
        raise ValueError(f"Could not get name for: {obj}")


class Inputs:
    args: tp.Tuple[tp.Any, ...]
    kwargs: tp.Dict[str, tp.Any]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_value(cls, value: InputLike) -> "Inputs":
        if isinstance(value, cls):
            return value
        elif isinstance(value, tuple):
            return cls(*value)
        elif isinstance(value, dict):
            return cls(**value)
        else:
            return cls(value)


def _to_str(p):
    return "".join(p.astype(int).reshape(-1).astype(str).tolist())


def format_predictions(pred, mask=None, threshold=0.5):
    """
    Args:
        pred: model output without batch dim
            instance_output: [n, patch_size, patch_size] float
            instance_yx, instance_xc: [n, patch_size, patch_size]
            pred_scores: [n]
            instance_mask: [n, 1, 1]
        mask: optional mask selecting cells
        threshold: float
    Returns:
        bboxes: [n, 4] int64
        encodings: [n] string
    """
    patches = np.asarray(pred["instance_output"]) > threshold
    yc = np.asarray(pred["instance_yc"])
    xc = np.asarray(pred["instance_xc"])
    scores = np.asarray(pred["pred_scores"])
    bboxes = np.asarray(bboxes_of_patches(pred))

    is_valid = pred["instance_mask"].squeeze(axis=(-1, -2))
    is_valid & patches.any(axis=(1, 2))  # no empty patches
    if mask is not None:
        is_valid &= mask

    patches = patches[is_valid]
    yc = yc[is_valid]
    xc = xc[is_valid]
    scores = scores[is_valid]

    encodings = []
    for (r0, c0, r1, c1), y0, x0, p in zip(bboxes, yc[:, 0, 0], xc[:, 0, 0], patches):
        roi = p[r0 - y0 : r1 - y0, c0 - x0 : c1 - x0]
        encodings.append(_to_str(roi))

    n_pixels = np.count_nonzero(patches, axis=(1, 2))
    yx = np.stack(
        [
            (patches * yc).sum(axis=(1, 2)) / n_pixels,
            (patches * xc).sum(axis=(1, 2)) / n_pixels,
        ],
        axis=-1,
    )  # centroid

    return bboxes, encodings, scores, yx
