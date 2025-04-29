import pytest
import pandas as pd
import numpy as np
from OutText_preprocessing.outlier_removal import OutlierRemover

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "A": [10, 12, 14, 16, 18, 1000],  # obvious outlier
        "B": [1, 1, 2, 2, 3, 3],
        "C": [100, 200, 300, 400, 500, 600]
    })

def test_zscore_removal(sample_data):
    remover = OutlierRemover(method="zscore", threshold=3)
    cleaned = remover.fit_transform(sample_data)
    assert len(cleaned) < len(sample_data)
    assert "A" in cleaned.columns

def test_zscore_capper(sample_data):
    remover = OutlierRemover(method="zscore_capper", threshold=2)
    capped = remover.fit_transform(sample_data)
    assert capped["A"].max() < sample_data["A"].max()
    assert capped.shape == sample_data.shape

def test_yeo_johnson_removal(sample_data):
    remover = OutlierRemover(method="yeo_johnson", threshold=2.5)
    transformed = remover.fit_transform(sample_data)
    assert len(transformed) < len(sample_data)

def test_yeo_johnson_capper(sample_data):
    remover = OutlierRemover(method="yeo_johnson_capper", threshold=2)
    capped = remover.fit_transform(sample_data)
    assert capped.shape == sample_data.shape
    assert np.all(capped.select_dtypes(include=[np.number]).max() <= 2)

def test_reduce_impact(sample_data):
    remover = OutlierRemover(method="impact_reduction", threshold=2)
    reduced = remover.fit_transform(sample_data)
    assert reduced["A"].max() < sample_data["A"].max()

def test_adaptive_trimming(sample_data):
    remover = OutlierRemover(method="adaptive_trimming")
    trimmed = remover.fit_transform(sample_data)
    assert trimmed.shape == sample_data.shape

def test_smooth_capping(sample_data):
    remover = OutlierRemover(method="smooth_capping", threshold=2)
    capped = remover.fit_transform(sample_data)
    assert capped.shape == sample_data.shape

def test_local_standardization(sample_data):
    remover = OutlierRemover(method="local_standardization", threshold=2)
    corrected = remover.fit_transform(sample_data)
    assert corrected.shape == sample_data.shape

def test_invalid_method(sample_data):
    with pytest.raises(ValueError):
        OutlierRemover(method="invalid").fit_transform(sample_data)

def test_multi_outlier_multi_columns(sample_data):
    remover = OutlierRemover(threshold=2)
    methods_dict = {
        "zscore_capper": ["A"],
        "impact_reduction": ["B", "C"]
    }
    transformed = remover.multi_outlier_multi_columns(sample_data, methods_dict)
    assert transformed.shape == sample_data.shape
    assert not transformed.equals(sample_data)

def test_repr():
    remover = OutlierRemover(method="zscore_capper", threshold=2.5)
    assert repr(remover) == "OutlierRemover(method=zscore_capper, threshold=2.5, smooth_factor=0.9)"
