import os
from pathlib import Path

import pandas as pd

from machine_learning.dataset_manager import DatasetManager, load_uci_adult_dataset, validate_csv_file


def test_validate_csv_file_accepts_valid_csv(tmp_path):
    path = tmp_path / 'sample.csv'
    pd.DataFrame({'age': [25, 30], 'income': ['>50K', '<=50K']}).to_csv(path, index=False)

    result = validate_csv_file(path)

    assert result is True


def test_dataset_manager_loads_csv(tmp_path):
    path = tmp_path / 'adult_like.csv'
    pd.DataFrame({
        'age': [25, 30, 45],
        'workclass': ['Private', 'Self-emp', 'Private'],
        'education': ['Bachelors', 'HS-grad', 'Masters'],
        'income': ['>50K', '<=50K', '>50K']
    }).to_csv(path, index=False)

    manager = DatasetManager(base_dir=str(tmp_path))
    dataset = manager.load_dataset(str(path), target_column='income')

    assert dataset is not None
    assert 'income' in dataset.columns
    assert len(dataset) == 3


def test_load_dataset_normalizes_adult_missing_values(tmp_path):
    path = tmp_path / 'adult_missing.csv'
    pd.DataFrame({
        'workclass': [' Private', ' ?'],
        'occupation': [' Sales ', ' ?'],
        'income': [' <=50K', ' >50K'],
    }).to_csv(path, index=False)

    from machine_learning.dataset_manager import load_dataset
    dataset = load_dataset(path, target_column='income')

    assert pd.isna(dataset.loc[1, 'workclass'])
    assert pd.isna(dataset.loc[1, 'occupation'])
    assert dataset.loc[0, 'income'] == '<=50K'


def test_official_uci_adult_files_load_with_original_partition():
    data_directory = Path(__file__).resolve().parents[1] / 'dataset'
    frame, training_indices, testing_indices = load_uci_adult_dataset(
        data_directory / 'adult.data',
        data_directory / 'adult.test',
    )

    assert len(training_indices) == 32561
    assert len(testing_indices) == 16281
    assert len(frame) == 48842
    assert len(frame.columns) == 15
    assert frame.columns[-1] == 'income'
    assert set(frame['income'].unique()) == {'<=50K', '>50K'}
    assert max(training_indices) < min(testing_indices)
