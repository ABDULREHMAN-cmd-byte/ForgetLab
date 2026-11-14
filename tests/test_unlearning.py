import pandas as pd

from machine_learning.unlearn import approximate_unlearning_experiment, exact_retraining_unlearning
from machine_learning.train import train_model


def test_exact_retraining_baseline_runs():
    df = pd.DataFrame({
        'age': [18, 21, 25, 29, 35, 40, 45, 50, 55, 60],
        'education': ['HS-grad', 'School', 'Bachelors', 'School', 'Masters', 'Bachelors', 'Masters', 'Doctorate', 'Bachelors', 'Masters'],
        'income': ['<=50K', '<=50K', '<=50K', '>50K', '>50K', '>50K', '>50K', '>50K', '>50K', '>50K']
    })

    result = exact_retraining_unlearning(
        df,
        target_column='income',
        forget_indices=[3, 4],
        model_name='logistic_regression',
    )

    assert 'original_metrics' in result
    assert 'unlearned_metrics' in result
    assert 'comparison' in result


def test_approximate_unlearning_experiment_runs():
    df = pd.DataFrame({
        'age': [20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
        'education': ['School', 'HS-grad', 'Bachelors', 'School', 'HS-grad', 'Bachelors', 'Masters', 'Doctorate', 'Masters', 'Bachelors'],
        'income': ['<=50K', '<=50K', '<=50K', '>50K', '>50K', '>50K', '>50K', '>50K', '>50K', '>50K']
    })

    result = approximate_unlearning_experiment(
        df,
        target_column='income',
        forget_indices=[3, 4],
        model_name='sgd_classifier',
    )

    assert 'original_metrics' in result
    assert 'approximate_metrics' in result
    assert 'notes' in result


def test_approximate_unlearning_rejects_unsupported_classifier(df=None):
    frame = pd.DataFrame({
        'age': [20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
        'income': ['<=50K', '<=50K', '<=50K', '>50K', '>50K', '>50K', '>50K', '>50K', '>50K', '>50K'],
    })

    import pytest
    with pytest.raises(ValueError, match='supported only for SGD Classifier'):
        approximate_unlearning_experiment(frame, 'income', [3], model_name='logistic_regression')


def test_forty_row_methods_use_the_same_held_out_test_records(forty_row_income_dataset):
    df = forty_row_income_dataset
    baseline = train_model('logistic_regression', df, target_column='income')
    forget_indices = baseline['X_train'].index[:2].tolist()

    exact = exact_retraining_unlearning(df, 'income', forget_indices)
    approximate = approximate_unlearning_experiment(df, 'income', forget_indices, model_name='sgd_classifier')

    assert len(df) == 40
    assert len(df.columns) == 6
    assert exact['original_evaluation']['test_indices'] == exact['unlearned_evaluation']['test_indices']
    assert approximate['original_evaluation']['test_indices'] == [int(index) for index in baseline['X_test'].index]
    assert set(forget_indices).isdisjoint(exact['original_evaluation']['test_indices'])
