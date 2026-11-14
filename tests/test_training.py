import pandas as pd

from machine_learning.train import train_model


def test_train_model_returns_pipeline_and_metrics():
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'workclass': ['Private', 'Private', 'Local-gov', 'State-gov', 'Private', 'Private', 'Self-emp', None],
        'education': ['Bachelors', 'HS-grad', 'Masters', 'Bachelors', 'Masters', 'School', 'Doctorate', 'Bachelors'],
        'income': ['<=50K', '<=50K', '>50K', '>50K', '>50K', '<=50K', '>50K', '>50K']
    })

    result = train_model('logistic_regression', df, target_column='income')

    assert 'model' in result
    assert 'metrics' in result
    assert result['metrics']['accuracy'] >= 0.0
    assert result['metrics']['f1'] >= 0.0
    assert result['roc_curve'] is not None
    assert result['feature_importance']
