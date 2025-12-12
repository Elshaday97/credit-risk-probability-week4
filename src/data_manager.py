import pandas as pd
from pathlib import Path

from scripts import handle_errors
from scripts.constants import (
    CLEAN_DATA_DIR,
    CLEAN_DATA_FILE_NAME,
    RAW_DATA_DIR,
    RAW_DATA_FILE_NAME,
)


class DataManager:
    def __init__(self):
        self.clean_data_dir = Path(CLEAN_DATA_DIR)
        self.raw_data_dir = Path(RAW_DATA_DIR)
        self.clean_data_file_name = Path(CLEAN_DATA_FILE_NAME)
        self.raw_data_file_name = Path(RAW_DATA_FILE_NAME)

    @handle_errors
    def load_csv(self, load_clean=False) -> pd.DataFrame:
        """
        Docstring for load_csv
        :param load_clean: Boolean indicating whether to load clean data or raw data
        :return: DataFrame containing the loaded data
        :rtype: pd.DataFrame
        """
        path = (
            self.clean_data_dir / self.clean_data_file_name
            if load_clean
            else self.raw_data_dir / self.raw_data_file_name
        )

        print(f"Loading {path}...")
        if not Path(path).exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(f"The CSV at {path} is empty. Please try again!")

        print(f"Sucessfully loaded {path}!")
        return df

    @handle_errors
    def clean_data(self):
        pass

    @handle_errors
    def save_to_csv(self, df: pd.DataFrame):
        """
        Docstring for save_to_csv
        :param df: DataFrame to be saved
        :type df: pd.DataFrame
        """
        if df.empty:
            raise ValueError("Dataframe is empty and can not be saved.")

        path = CLEAN_DATA_DIR / CLEAN_DATA_FILE_NAME
        if not Path(path).exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        df.to_csv(path)
        print(f"Sucessfully saved dataframe to {path}!")
