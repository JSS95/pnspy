from pns import pns
from pns.util import circular_data


def test_pns_dtype():
    x = circular_data().astype("float32")
    vs, rs, xis, x_transform = pns(x, 1)
    assert all([v.dtype == x.dtype for v in vs])
    assert rs.dtype == x.dtype
    assert xis.dtype == x.dtype
    assert x_transform.dtype == x.dtype
