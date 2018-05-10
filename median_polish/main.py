from typing import Dict
import numpy as np

def median_polish(data: np.ndarray, n_iter: int = 10) -> Dict[str, np.ndarray]:
    """Performs median polish on a 2-D array
    Args:
        data: input 2-D array
    Returns:
        a dict, with:
            ave: μ
            col: column effect
            row: row effect
            r: cell residue
    """
    assert data.ndim == 2, "Input must be 2D array"
    ndim = 2
    data = data.copy()
    grand_effect = np.median(data)
    data -= grand_effect
    median_margins = [0] * ndim
    margins = [np.zeros(shape=data.shape[idx]) for idx in range(2)]
    dim_mask = np.ones(ndim, dtype=np.int)

    for _ in range(n_iter):
        for dim_id in range(ndim):
            rest_dim = 1 - dim_id
            temp_median = np.median(data, rest_dim)
            margins[dim_id] += temp_median
            median_margins[rest_dim] = np.median(margins[rest_dim])
            margins[rest_dim] -= median_margins[rest_dim]
            dim_mask[dim_id] = -1
            data -= temp_median.reshape(dim_mask)
            dim_mask[dim_id] = 1
        grand_effect += sum(median_margins)
    return {'ave': grand_effect, 'row': margins[1], 'column': margins[0], 'r': data}


def med_abs_dev(data: np.ndarray) -> float:
    """Median absolute deviation.
    MAD = median(|X_i - median(X)|)
    """
    return np.median(np.abs(data - np.median(data)))
