import warnings

import numpy as np

from pns import pns
from pns.pss import pss
from pns.util import circular_data


def test_pns_maxiter_warning():
    X = circular_data([0, -1, 0])
    with warnings.catch_warnings(record=True) as w:
        pns(X, 1, maxiter=1)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)


def test_pss_zero_norm_fallback():
    """Test that pss() handles zero norm case when D=2."""
    # Create data on opposite sides of a circle (zero mean)
    x = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    _, r = pss(x)  # Does not fail
