import pandas as pd
from src.data import loader, saver
import os

def test_load_local_csv(tmp_path):
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    test_file = tmp_path / "test.csv"
    df.to_csv(test_file, index=False)

    loaded_df = loader.load_local_file(str(test_file))
    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.equals(df)

def test_save_dataset(tmp_path):
    df = pd.DataFrame({'X': [1, 2, 3]})
    filepath = saver.save_dataset(df, tmp_path, name="test_save")
    assert os.path.exists(filepath)
    saved_df = pd.read_csv(filepath)
    assert saved_df.equals(df)
