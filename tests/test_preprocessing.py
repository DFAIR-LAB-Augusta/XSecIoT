
import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from src.FIRE.preprocessing import (
    _entropy,
    clean_data,
)


# Fixture for temporary CSV files\@pytest.fixture
def tmp_csv(tmp_path, request):
    df = request.param
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


# Sample default dataset
DEFAULT_DF = pd.DataFrame({
    'src_ip': ['1', '1', '2'],
    'dst_ip': ['a', 'a', 'b'],
    'src_port': [1000, 1000, 2000],
    'dst_port': [80, 80, 443],
    'protocol': [6, 6, 17],
    'timestamp': ['01-01-2020 00:00', '01-01-2020 00:01', '01-01-2020 00:02'],
    'flow_duration': [1.0, 2.0, 3.0],
    'tot_fwd_pkt': [1, 2, 3],
    'tot_bwd_pkts': [0, 1, 0],
    'totlen_fwd_pkts': [100, 200, 300],
    'totlen_bwd_pkts': [0, 100, 0]
})

# Sample UNSW dataset
UNSW_DF = pd.DataFrame({
    'IPV4_SRC_ADDR': ['1'],
    'IPV4_DST_ADDR': ['a'],
    'L4_SRC_PORT': [1000],
    'L4_DST_PORT': [80],
    'PROTOCOL': [6],
    'FLOW_START_MILLISECONDS': [0],
    'FLOW_END_MILLISECONDS': [1000],
    'FLOW_DURATION_MILLISECONDS': [1],
    'IN_PKTS': [1],
    'OUT_PKTS': [0],
    'IN_BYTES': [0],
    'OUT_BYTES': [100],
    'SRC_TO_DST_IAT_MIN': [0],
    'SRC_TO_DST_IAT_MAX': [1],
    'SRC_TO_DST_IAT_AVG': [0.5],
    'SRC_TO_DST_IAT_STDDEV': [0.1],
    'DST_TO_SRC_IAT_MIN': [0],
    'DST_TO_SRC_IAT_MAX': [1],
    'DST_TO_SRC_IAT_AVG': [0.5],
    'DST_TO_SRC_IAT_STDDEV': [0.1]
})


# Tests for _clean_data
@pytest.mark.parametrize("df, is_unsw", [
    (DEFAULT_DF.copy(), False),
    (UNSW_DF.copy(), True)
])
def test_clean_data(df, is_unsw):
    cleaned = clean_data(df, is_unsw)
    # Should set index
    assert hasattr(cleaned.index, 'min')
    # Should not contain infinities
    assert not np.isinf(cleaned.select_dtypes(include=[np.number])).any().any()


# Test entropy
def test_entropy_uniform():
    data = pd.Series([1, 2, 3, 4])
    ent = _entropy(data)
    assert pytest.approx(ent) == stats.entropy(data.value_counts(normalize=True))

