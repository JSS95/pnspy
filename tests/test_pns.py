import warnings

import numpy as np

from pns import (
    extrinsic_transform,
    intrinsic_transform,
    inverse_extrinsic_transform,
    inverse_intrinsic_transform,
    pns,
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
    X = circular_data([0, -1, 0])
    vs, rs, _, _ = pns(X, 1)
    X_transform = intrinsic_transform(X, vs, rs)
    Xinv = inverse_intrinsic_transform(X_transform, vs, rs)
    assert np.all(np.isclose(X, Xinv, atol=1e-1))


def test_inverse_extrinsic_transform():
    X = circular_data([0, -1, 0])
    vs, rs, _, _ = pns(X, 2)
    X_transform = extrinsic_transform(X, vs, rs)
    Xinv = inverse_extrinsic_transform(X_transform, vs, rs)
    X_transform2 = extrinsic_transform(Xinv, vs, rs)
    assert np.all(np.isclose(X_transform, X_transform2, atol=1e-1))
