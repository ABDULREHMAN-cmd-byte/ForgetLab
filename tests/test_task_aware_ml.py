import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from machine_learning.preprocessing import build_preprocessing_pipeline, detect_target_task, get_supported_models
from machine_learning.train import split_training_data, train_model
from machine_learning.unlearn import approximate_unlearning_experiment, exact_retraining_unlearning


ADULT_TARGET_TASKS = {
    "income": "binary_classification",
    "age": "regression",
    "workclass": "multiclass_classification",
    "education": "multiclass_classification",
    "occupation": "multiclass_classification",
    "hours-per-week": "regression",
    "education-num": "regression",
    "marital-status": "multiclass_classification",
    "capital-gain": "regression",
    "native-country": "multiclass_classification",
}


@pytest.mark.parametrize("target_column,expected_task", ADULT_TARGET_TASKS.items())
def test_original_adult_target_type_detection(official_uci_adult_dataset, target_column, expected_task):
    frame, _, _ = official_uci_adult_dataset
    assert detect_target_task(frame[target_column]) == expected_task
    expected_family = "regressor" if expected_task == "regression" else "classifier"
    assert all(expected_family in model["key"] or model["key"] in {"linear_regression", "logistic_regression"}
               for model in get_supported_models(expected_task))


def test_income_binary_training_uses_only_original_train_and_test(official_uci_adult_dataset):
    frame, official_train, official_test = official_uci_adult_dataset
    result = train_model(
        "logistic_regression", frame, "income",
        train_indices=official_train, test_indices=official_test,
    )

    assert result["task_type"] == "binary_classification"
    assert set(result["X_train"].index).issubset(set(official_train))
    assert set(result["X_test"].index) == set(official_test)
    assert "income" not in result["X_train"].columns
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} == set(result["metrics"])
    assert result["confusion_matrix"] is not None
    assert result["precision_recall_curve"] is not None


def test_adult_workclass_multiclass_metrics(official_uci_adult_dataset):
    frame, official_train, official_test = official_uci_adult_dataset
    result = train_model(
        "logistic_regression", frame, "workclass",
        train_indices=official_train, test_indices=official_test,
    )

    assert result["task_type"] == "multiclass_classification"
    assert len(result["class_labels"]) > 2
    assert result["confusion_matrix"] is not None
    assert result["precision_recall_curve"] is None
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} == set(result["metrics"])
    assert len(result["X_test"]) < len(official_test)
    assert set(result["X_test"].index).issubset(set(official_test))


def test_adult_age_regression_metrics_and_plots(official_uci_adult_dataset):
    frame, official_train, official_test = official_uci_adult_dataset
    result = train_model(
        "linear_regression", frame, "age",
        train_indices=official_train, test_indices=official_test,
    )

    assert result["task_type"] == "regression"
    assert set(result["metrics"]) == {"mae", "mse", "rmse", "r2"}
    assert result["actual_vs_predicted"] is not None
    assert result["residuals"] is not None
    assert len(result["actual_vs_predicted"]["actual"]) == len(result["X_test"])
    assert set(result["X_test"].index).issubset(set(official_test))


def test_exact_regression_retraining_keeps_held_out_test_fixed(official_uci_adult_dataset):
    frame, official_train, official_test = official_uci_adult_dataset
    selected = official_train[:2]
    result = exact_retraining_unlearning(
        frame, "age", selected, model_name="linear_regression",
        train_indices=official_train, test_indices=official_test,
    )

    assert result["task_type"] == "regression"
    assert result["original_evaluation"]["test_indices"] == result["unlearned_evaluation"]["test_indices"]
    assert set(selected).isdisjoint(result["original_evaluation"]["test_indices"])
    assert set(result["unlearned_metrics"]) == {"mae", "mse", "rmse", "r2"}


def test_approximate_sgd_support_is_task_and_estimator_specific(forty_row_income_dataset):
    frame = forty_row_income_dataset
    X_train, _, _, _ = split_training_data(frame, "income")
    selected = X_train.index[:2].tolist()
    classifier_result = approximate_unlearning_experiment(
        frame, "income", selected, model_name="sgd_classifier",
    )
    assert classifier_result["task_type"] == "binary_classification"
    assert "accuracy" in classifier_result["approximate_metrics"]

    X_train, _, _, _ = split_training_data(frame, "age")
    selected = X_train.index[:2].tolist()
    regressor_result = approximate_unlearning_experiment(
        frame, "age", selected, model_name="sgd_regressor",
    )
    assert regressor_result["task_type"] == "regression"
    assert set(regressor_result["approximate_metrics"]) == {"mae", "mse", "rmse", "r2"}

    with pytest.raises(ValueError, match="supported only for SGD Classifier"):
        approximate_unlearning_experiment(frame, "income", selected, model_name="logistic_regression")


def test_preprocessing_handles_missing_and_unseen_categories_without_target_leakage():
    train = pd.DataFrame({"region": ["north", "south", None, "north"], "amount": [1, 2, 3, 4]})
    test = pd.DataFrame({"region": ["never-seen"], "amount": [5]})
    preprocessor = build_preprocessing_pipeline(train)
    transformed_train = preprocessor.fit_transform(train)
    transformed_test = preprocessor.transform(test)

    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert np.isfinite(transformed_test.toarray() if hasattr(transformed_test, "toarray") else transformed_test).all()
    frame = pd.DataFrame({"region": ["north", "south", "north", "south"], "target": [1, 2, 3, 4]})
    X_train, _, _, _ = split_training_data(frame, "target")
    assert "target" not in X_train.columns
