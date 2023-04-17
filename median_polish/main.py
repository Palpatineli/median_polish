from typing import Dict, cast, List
from numpy import ndarray, ones, zeros, float_, abs, median  # type: ignore
from numpy.typing import NDArray

def median_polish(data: ndarray, n_iter: int = 10) -> Dict[str, float | NDArray[float_]]:
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
    grand_effect = cast(float, median(data))
    data -= grand_effect
    median_margins: List[float] = [0] * ndim
    margins = [zeros(shape=data.shape[idx]) for idx in range(2)]
    dim_mask = ones(ndim, dtype=int)

    for _ in range(n_iter):
        for dim_id in range(ndim):
            rest_dim = 1 - dim_id
            temp_median = cast(NDArray[float_], median(data, rest_dim))
            margins[dim_id] += temp_median
            median_margins[rest_dim] = cast(float, median(margins[rest_dim]))
            margins[rest_dim] -= median_margins[rest_dim]
            dim_mask[dim_id] = -1
            data -= temp_median.reshape(dim_mask)
            dim_mask[dim_id] = 1
        grand_effect += sum(median_margins)
    return {'ave': grand_effect, 'row': margins[1], 'column': margins[0], 'r': data}


def med_abs_dev(data: ndarray) -> float:
    """Median absolute deviation.
    MAD = median(|X_i - median(X)|)
    """
    return cast(float, median(abs(data - median(data))))
