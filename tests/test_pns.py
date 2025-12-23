import warnings

import numpy as np

from pns import (
    fit_transform,
    inverse_transform,
    pns,
    transform,
)
from pns.pss import pss
from pns.util import circular_data


def test_pns_maxiter_warning():
    X = circular_data([0, -1, 0])
    with warnings.catch_warnings(record=True) as w:
        pns(X, 1, maxiter=1)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)


def test_pss_zero_norm_():
    """Test that pss() handles zero norm case when D=2."""
    # Create data on opposite sides of a circle (zero mean)
    x = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    pss(x)  # Does not fail


def test_inverse_intrinsic_transform():
    n = 1
    X = circular_data([0, -1, 0])
    vs, rs, x_transformed = fit_transform(X, n, type="intrinsic")
    Xinv = inverse_transform(x_transformed, vs, rs, type="intrinsic")
    X_transform2 = transform(Xinv, vs, rs, n_components=n, type="intrinsic")
    assert np.all(np.isclose(x_transformed, X_transform2, atol=1e-1))


def test_inverse_extrinsic_transform():
    n = 2
    X = circular_data([0, -1, 0])
    vs, rs, x_transformed = fit_transform(X, n, type="extrinsic")
    Xinv = inverse_transform(x_transformed, vs, rs, type="extrinsic")
    X_transform2 = transform(Xinv, vs, rs, n_components=n, type="extrinsic")
    assert np.all(np.isclose(x_transformed, X_transform2, atol=1e-1))
