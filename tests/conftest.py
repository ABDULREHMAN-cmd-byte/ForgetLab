import pandas as pd
import pytest
from pathlib import Path

from machine_learning.dataset_manager import load_uci_adult_dataset

@pytest.fixture
def forty_row_income_dataset():
    """The uploaded 40-row, six-column fixture used for unlearning regressions."""
    row_count = 40
    return pd.DataFrame({
        "age": [20 + index for index in range(row_count)],
        "workclass": ["Private" if index % 3 else "Government" for index in range(row_count)],
        "education": ["Bachelors" if index % 2 else "HS-grad" for index in range(row_count)],
        "has_degree": [index % 2 for index in range(row_count)],
        "annual_income": [35000 + (index % 8) * 7000 for index in range(row_count)],
        "income": ["<=50K" if index % 2 == 0 else ">50K" for index in range(row_count)],
    })


@pytest.fixture(scope="session")
def official_uci_adult_dataset():
    """Load the checked-in official UCI pair without downloading or rewriting it."""
    path = Path(__file__).resolve().parents[1] / "dataset"
    return load_uci_adult_dataset(path / "adult.data", path / "adult.test")