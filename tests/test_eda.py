import pandas as pd
from src.eda import eda1_raw, eda2_processed

df = pd.DataFrame({
    "num": [1, 2, 3, 4],
    "cat": ["A", "B", "A", "C"]
})

def test_eda1_raw_summary():
    summary = eda1_raw.generate_summary(df)
    assert isinstance(summary, dict)
    assert "shape" in summary

def test_eda2_processed_summary():
    summary = eda2_processed.generate_summary(df)
    assert isinstance(summary, dict)
    assert "missing_values" in summary
