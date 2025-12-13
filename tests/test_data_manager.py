import pandas as pd
import pytest
from src import DataManager


@pytest.fixture
def sample_df():
    """Create a small sample DataFrame for testing."""
    return pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})


@pytest.fixture
def data_manager(tmp_path, monkeypatch):
    """
    Create a DataManager instance with temporary paths.
    """
    # 1. Setup temporary directories
    raw_dir = tmp_path / "raw"
    clean_dir = tmp_path / "clean"
    raw_dir.mkdir()
    clean_dir.mkdir()

    # 2. Create dummy CSV file for loading tests
    raw_file = raw_dir / "raw.csv"
    raw_file.write_text("A,B\n1,x\n2,y")

    # 3. Patch the constants used inside DataManager
    # NOTE: If DataManager imports constants like `from scripts.constants import RAW_DATA_DIR`,
    # you might need to patch 'src.data_manager.RAW_DATA_DIR' instead.
    # Assuming DataManager uses `scripts.constants.RAW_DATA_DIR` directly:
    monkeypatch.setattr("src.data_manager.RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr("src.data_manager.CLEAN_DATA_DIR", clean_dir)
    monkeypatch.setattr("src.data_manager.RAW_DATA_FILE_NAME", "raw.csv")
    monkeypatch.setattr("src.data_manager.CLEAN_DATA_FILE_NAME", "clean.csv")

    return DataManager()


def test_load_raw_csv_success(data_manager):
    """Test loading a raw CSV file successfully."""
    df = data_manager.load_csv(load_clean=False)

    assert not df.empty
    assert list(df.columns) == ["A", "B"]
    assert len(df) == 2


def test_load_csv_file_not_found(data_manager):
    """Test that FileNotFoundError is raised when file is missing."""
    # Try to load 'clean' data which we haven't created yet
    with pytest.raises(FileNotFoundError):
        data_manager.load_csv(load_clean=True)


def test_save_to_csv_empty_dataframe_raises(data_manager):
    """Test that saving an empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError):
        data_manager.save_to_csv(empty_df)
