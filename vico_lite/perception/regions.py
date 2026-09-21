from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

MIN_PIXELS = 4
PAD_RATIO = 0.15
MIN_SIDE = 16


def _bounding_box(mask: np.ndarray, pad_ratio: float) -> Optional[Tuple[int, int, int, int]]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1

    height, width = mask.shape
    pad_y = max(int(round((bottom - top) * pad_ratio)), 2)
    pad_x = max(int(round((right - left) * pad_ratio)), 2)
    top = max(top - pad_y, 0)
    bottom = min(bottom + pad_y, height)
    left = max(left - pad_x, 0)
    right = min(right + pad_x, width)

    # CLIP downsamples aggressively, so grow very thin boxes around their centre.
    if bottom - top < MIN_SIDE:
        centre = (top + bottom) // 2
        top = max(centre - MIN_SIDE // 2, 0)
        bottom = min(top + MIN_SIDE, height)
    if right - left < MIN_SIDE:
        centre = (left + right) // 2
        left = max(centre - MIN_SIDE // 2, 0)
        right = min(left + MIN_SIDE, width)
    if bottom <= top or right <= left:
        return None
    return top, bottom, left, right


def extract_object_patches(
    rgb: np.ndarray,
    seg_mask: Optional[np.ndarray],
    object_infos: Sequence[Dict[str, Any]],
) -> Tuple[List[np.ndarray], List[int]]:
    """Crop one image patch per segmented object.

    Returns the patches together with the index of the object each patch came
    from. Objects whose segmentation colour covers fewer than ``MIN_PIXELS``
    pixels are dropped: they carry too little signal for a meaningful CLIP
    embedding.
    """
    if seg_mask is None or rgb is None:
        return [], []
    seg = np.asarray(seg_mask)
    if seg.ndim != 3 or seg.shape[2] < 3:
        return [], []

    patches: List[np.ndarray] = []
    indices: List[int] = []
    for index, info in enumerate(object_infos):
        colour = info.get("seg_color")
        if colour is None:
            continue
        try:
            colour_array = np.asarray(colour, dtype=seg.dtype).reshape(1, 1, -1)[:, :, :3]
        except (TypeError, ValueError):
            continue
        mask = np.all(seg[:, :, :3] == colour_array, axis=-1)
        if int(mask.sum()) < MIN_PIXELS:
            continue
        box = _bounding_box(mask, PAD_RATIO)
        if box is None:
            continue
        top, bottom, left, right = box
        patches.append(rgb[top:bottom, left:right])
        indices.append(index)
    return patches, indices


__all__ = ["extract_object_patches"]
