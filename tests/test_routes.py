from io import BytesIO

import app as app_module
from app import app_state, create_app
from config import CONFIG


def _isolate_app_state(monkeypatch):
    defaults = {
        'dataset_path': None,
        'dataset_key': None,
        'dataset_name': None,
        'dataset_source': None,
        'dataframe': None,
        'target_column': 'income',
        'split_indices': None,
        'partition_indices': None,
        'original_result': None,
        'unlearned_result': None,
        'unlearning_results': {},
        'forget_indices': [],
        'training_time': 0.0,
        'unlearning_time': 0.0,
        'unlearning_method': None,
    }
    for key, value in defaults.items():
        monkeypatch.setitem(app_state, key, value)


def test_app_starts_and_dashboard_routes():
    app = create_app()
    client = app.test_client()

    response = client.get('/')
    assert response.status_code == 200

    response = client.get('/health')
    assert response.status_code == 200
    assert b'ok' in response.data.lower()


def test_navigation_pages_and_dashboard_chart_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(CONFIG, 'SQLALCHEMY_DATABASE_URI', f"sqlite:///{tmp_path / 'navigation.db'}")
    app = create_app()
    client = app.test_client()
    for path in [
        '/', '/datasets', '/model-explorer', '/unlearning', '/evaluation',
        '/history', '/reports', '/documentation',
    ]:
        assert client.get(path).status_code == 200

    payload = client.get('/api/dashboard').get_json()
    assert payload['dataset']['rows'] > 0
    assert payload['has_model'] is False
    assert {
        'class_distribution', 'missing_values', 'correlation', 'confusion_matrix',
        'roc', 'precision_recall', 'feature_importance', 'forgetting_records',
        'method_comparison', 'timings',
    }.issubset(payload['charts'])
    assert client.get('/api/report/export/json').status_code == 409
    compact = client.get('/api/dashboard?view=unlearning').get_json()
    assert {
        'method_comparison', 'comparison_metrics', 'forgetting_records', 'timings',
    } == set(compact['charts'])


def test_fresh_app_defaults_to_official_uci_dataset(monkeypatch):
    defaults = {
        'dataset_path': None,
        'dataset_key': None,
        'dataset_name': None,
        'dataset_source': None,
        'dataframe': None,
        'target_column': 'income',
        'split_indices': None,
        'partition_indices': None,
        'original_result': None,
        'unlearned_result': None,
        'unlearning_results': {},
        'forget_indices': [],
        'training_time': 0.0,
        'unlearning_time': 0.0,
    }
    for key, value in defaults.items():
        monkeypatch.setitem(app_state, key, value)

    client = create_app().test_client()
    active = client.get('/api/dataset').get_json()
    catalog = client.get('/api/datasets').get_json()

    assert active['dataset_key'] == 'uci_adult'
    assert active['name'] == 'UCI Adult Census Income'
    assert active['source'].startswith('UCI Machine Learning Repository')
    assert active['rows'] == 48842
    assert active['training_rows'] == 32561
    assert active['testing_rows'] == 16281
    assert active['target_column'] == 'income'
    assert any(item['key'] == 'upload:forgetlab_test_dataset.csv' for item in catalog['datasets'])


def test_uci_target_switches_preserve_official_partition(monkeypatch):
    reset_values = {
        'dataset_path': None,
        'dataset_key': None,
        'dataset_name': None,
        'dataset_source': None,
        'dataframe': None,
        'target_column': 'income',
        'split_indices': None,
        'partition_indices': None,
        'original_result': None,
        'unlearned_result': None,
        'unlearning_results': {},
        'forget_indices': [],
        'training_time': 0.0,
        'unlearning_time': 0.0,
    }
    for key, value in reset_values.items():
        monkeypatch.setitem(app_state, key, value)

    client = create_app().test_client()
    expected_tasks = {
        'income': 'binary_classification',
        'workclass': 'multiclass_classification',
        'education': 'multiclass_classification',
        'occupation': 'multiclass_classification',
        'hours-per-week': 'regression',
        'education-num': 'regression',
        'marital-status': 'multiclass_classification',
        'capital-gain': 'regression',
        'native-country': 'multiclass_classification',
    }
    official_train = set(range(32561))
    official_test = set(range(32561, 48842))
    for target, task_type in expected_tasks.items():
        response = client.post('/api/target', json={'target_column': target})
        assert response.status_code == 200
        assert response.get_json()['task_type'] == task_type
        dataset = client.get('/api/dataset').get_json()
        usable_count = int(app_state['dataframe'][target].notna().sum())
        assert dataset['training_rows'] + dataset['testing_rows'] == usable_count
        assert set(app_state['split_indices']['train']).issubset(official_train)
        assert set(app_state['split_indices']['test']).issubset(official_test)


def test_dataset_activation_preserves_uploaded_csv_as_optional(
    forty_row_income_dataset, monkeypatch, tmp_path
):
    monkeypatch.setattr(CONFIG, 'SQLALCHEMY_DATABASE_URI', f"sqlite:///{tmp_path / 'catalog.db'}")
    monkeypatch.setattr(CONFIG, 'UPLOAD_FOLDER', str(tmp_path / 'uploads'))
    client = create_app().test_client()
    csv_contents = forty_row_income_dataset.to_csv(index=False).encode('utf-8')
    upload = client.post(
        '/api/dataset/upload',
        data={'file': (BytesIO(csv_contents), 'forty_rows.csv')},
        content_type='multipart/form-data',
    )
    assert upload.status_code == 200

    official = client.post('/api/dataset/activate', json={'dataset_key': 'uci_adult'})
    assert official.status_code == 200
    assert official.get_json()['dataset']['rows'] == 48842
    assert official.get_json()['dataset']['testing_rows'] == 16281

    uploaded = client.post('/api/dataset/activate', json={'dataset_key': 'upload:forty_rows.csv'})
    assert uploaded.status_code == 200
    assert uploaded.get_json()['dataset']['rows'] == 40
    assert uploaded.get_json()['dataset']['target_column'] == 'income'
    assert (tmp_path / 'uploads' / 'forty_rows.csv').is_file()


def test_unlearning_api_validates_forty_row_training_selection(
    forty_row_income_dataset, monkeypatch, tmp_path
):
    database_path = tmp_path / 'unlearning-test.db'
    upload_path = tmp_path / 'uploads'
    monkeypatch.setattr(CONFIG, 'SQLALCHEMY_DATABASE_URI', f'sqlite:///{database_path}')
    monkeypatch.setattr(CONFIG, 'UPLOAD_FOLDER', str(upload_path))
    app = create_app()
    client = app.test_client()
    csv_contents = forty_row_income_dataset.to_csv(index=False).encode('utf-8')

    response = client.post(
        '/api/dataset/upload',
        data={'file': (BytesIO(csv_contents), 'forty_rows.csv')},
        content_type='multipart/form-data',
    )
    assert response.status_code == 200
    active_dataset = client.get('/api/dataset').get_json()
    assert active_dataset['rows'] == 40
    assert len(active_dataset['columns']) == 6
    assert active_dataset['target_column'] == 'income'
    assert active_dataset['target_column'] != 'has_degree'

    selection = client.get('/api/unlearning/selection?offset=0&limit=50').get_json()
    training_indices = [row['index'] for row in selection['rows']]
    assert selection['training_count'] == len(training_indices)
    assert selection['testing_count'] == 8
    assert len(set(training_indices)) == 32
    assert set(training_indices).issubset(set(range(40)))
    held_out_indices = set(range(40)) - set(training_indices)
    selected_indices = training_indices[:2]

    for invalid_indices in [[], [-1], [100], list(range(100))]:
        invalid = client.post('/api/unlearn', json={
            'method': 'exact',
            'target_column': 'income',
            'forget_indices': invalid_indices,
        })
        assert invalid.status_code == 400
        assert invalid.get_json()['error']

    excessive_count = client.post('/api/unlearn', json={
        'method': 'exact',
        'target_column': 'income',
        'forget_count': 100,
        'forget_indices': training_indices[:2],
    })
    assert excessive_count.status_code == 400

    held_out = client.post('/api/unlearn', json={
        'method': 'exact',
        'target_column': 'income',
        'forget_indices': [min(held_out_indices)],
    })
    assert held_out.status_code == 400

    wrong_target = client.post('/api/unlearn', json={
        'method': 'exact',
        'target_column': 'has_degree',
        'forget_indices': selected_indices,
    })
    assert wrong_target.status_code == 409

    fixed_test_indices = None
    for method in ['exact', 'approximate']:
        result = client.post('/api/unlearn', json={
            'method': method,
            'target_column': 'income',
            'model_name': 'sgd_classifier' if method == 'approximate' else 'logistic_regression',
            'forget_indices': selected_indices,
        })
        assert result.status_code == 200, result.get_json()
        assert result.get_json()['status'] == 'ok'
        assert result.get_json()['request_id']
        assert len(result.data) < 20000
        original_evaluation = result.get_json()['result']['original_evaluation']
        assert original_evaluation['testing_count'] == 8
        assert original_evaluation['training_count'] == 32
        if fixed_test_indices is not None:
            assert original_evaluation['test_index_digest'] == fixed_test_indices
        fixed_test_indices = original_evaluation['test_index_digest']
        if method == 'exact':
            assert original_evaluation['test_index_digest'] == result.get_json()['result']['unlearned_evaluation']['test_index_digest']

    latest_result = result.get_json()['result']
    unlearning_dashboard = client.get('/api/dashboard?view=unlearning').get_json()
    assert unlearning_dashboard['stats']['unlearning_model_name'] == 'sgd_classifier'
    assert unlearning_dashboard['stats']['unlearning_original_metrics'] == latest_result['original_metrics']
    comparison = unlearning_dashboard['charts']['comparison_metrics']
    assert comparison['original'] == [
        latest_result['original_metrics'][key]
        for key in ['accuracy', 'precision', 'recall', 'f1']
    ]

    history = client.get('/api/history').get_json()['history']
    unlearning_history = [record for record in history if record['method'] in {'exact', 'approximate'}]
    assert {record['method'] for record in unlearning_history} == {'exact', 'approximate'}
    assert all(record['forget_count'] == 2 for record in unlearning_history)
    assert all(record['selected_indices'] == selected_indices for record in unlearning_history)
    history_count = len(unlearning_history)
    client.get('/api/dashboard?view=unlearning')
    client.get('/api/dashboard?view=unlearning')
    refreshed_history = client.get('/api/history').get_json()['history']
    assert len([record for record in refreshed_history if record['method'] in {'exact', 'approximate'}]) == history_count

    def fail_exact_unlearning(*args, **kwargs):
        raise RuntimeError('deliberate model failure')

    monkeypatch.setattr(app_module, 'exact_retraining_unlearning', fail_exact_unlearning)
    failed = client.post('/api/unlearn', json={
        'method': 'exact',
        'target_column': 'income',
        'forget_indices': training_indices[:1],
    })
    assert failed.status_code == 500
    assert failed.is_json
    assert failed.get_json()['request_id']
    assert 'failed during exact model computation' in failed.get_json()['error']


def test_dashboard_persists_training_and_unlearning_analytics(
    forty_row_income_dataset, monkeypatch, tmp_path
):
    _isolate_app_state(monkeypatch)
    monkeypatch.setattr(CONFIG, 'SQLALCHEMY_DATABASE_URI', f"sqlite:///{tmp_path / 'analytics.db'}")
    monkeypatch.setattr(CONFIG, 'UPLOAD_FOLDER', str(tmp_path / 'uploads'))
    monkeypatch.setattr(CONFIG, 'MODEL_ORIGINAL_FOLDER', str(tmp_path / 'models' / 'original'))
    monkeypatch.setattr(CONFIG, 'MODEL_UNLEARNED_FOLDER', str(tmp_path / 'models' / 'unlearned'))
    client = create_app().test_client()

    upload = client.post(
        '/api/dataset/upload',
        data={'file': (BytesIO(forty_row_income_dataset.to_csv(index=False).encode()), 'analytics.csv')},
        content_type='multipart/form-data',
    )
    assert upload.status_code == 200
    training = client.post('/api/train', json={
        'model_name': 'logistic_regression',
        'target_column': 'income',
    })
    assert training.status_code == 200

    dashboard = client.get('/api/dashboard').get_json()
    assert dashboard['task_type'] == 'binary_classification'
    assert dashboard['has_model'] is True
    assert len(dashboard['charts']['confusion_matrix']['values']) == 2
    assert dashboard['charts']['comparison_metrics']['original']

    selection = client.get('/api/unlearning/selection?limit=50').get_json()
    forgotten = [row['index'] for row in selection['rows'][:2]]
    unlearned = client.post('/api/unlearn', json={
        'method': 'exact',
        'target_column': 'income',
        'model_name': 'logistic_regression',
        'forget_indices': forgotten,
    })
    assert unlearned.status_code == 200

    compact = client.get('/api/dashboard?view=unlearning').get_json()
    assert compact['charts']['comparison_metrics']['labels'] == [
        'ACCURACY', 'PRECISION', 'RECALL', 'F1',
    ]
    assert compact['unlearning_history'][0]['method'] == 'exact'
    assert compact['unlearning_history'][0]['remaining_training_records'] == selection['training_count'] - 2

    app_state['original_result'] = None
    app_state['unlearned_result'] = None
    app_state['unlearning_results'] = {}
    app_state['forget_indices'] = []
    app_state['training_time'] = 0.0
    app_state['unlearning_time'] = 0.0
    app_state['unlearning_method'] = None
    refreshed = client.get('/api/dashboard').get_json()
    assert refreshed['has_model'] is True
    assert refreshed['has_unlearning'] is True
    assert refreshed['stats']['metrics']['accuracy'] == training.get_json()['metrics']['accuracy']
    assert refreshed['stats']['unlearned_metrics']['accuracy'] == unlearned.get_json()['result']['unlearned_metrics']['accuracy']
    assert refreshed['charts']['confusion_matrix']['values']
    assert client.get('/api/dashboard?view=unlearning').get_json()['unlearning_history'][0]['method'] == 'exact'
    restored_model = client.get('/api/model-explorer').get_json()
    assert restored_model['status'] == 'ok'
    assert restored_model['metrics']['accuracy'] == training.get_json()['metrics']['accuracy']


def test_dashboard_returns_regression_charts_without_classification_metrics(
    forty_row_income_dataset, monkeypatch, tmp_path
):
    _isolate_app_state(monkeypatch)
    monkeypatch.setattr(CONFIG, 'SQLALCHEMY_DATABASE_URI', f"sqlite:///{tmp_path / 'regression.db'}")
    monkeypatch.setattr(CONFIG, 'UPLOAD_FOLDER', str(tmp_path / 'uploads'))
    monkeypatch.setattr(CONFIG, 'MODEL_ORIGINAL_FOLDER', str(tmp_path / 'models' / 'original'))
    frame = forty_row_income_dataset.copy()
    frame['continuous_target'] = [float(1000 + index * 137.25) for index in range(len(frame))]
    client = create_app().test_client()

    upload = client.post(
        '/api/dataset/upload',
        data={'file': (BytesIO(frame.to_csv(index=False).encode()), 'regression.csv')},
        content_type='multipart/form-data',
    )
    assert upload.status_code == 200
    target_response = client.post('/api/target', json={'target_column': 'continuous_target'})
    assert target_response.status_code == 200
    trained = client.post('/api/train', json={
        'model_name': 'linear_regression',
        'target_column': 'continuous_target',
    })
    assert trained.status_code == 200

    dashboard = client.get('/api/dashboard').get_json()
    assert dashboard['task_type'] == 'regression'
    assert set(dashboard['stats']['metrics']) == {'mae', 'mse', 'rmse', 'r2'}
    assert len(dashboard['charts']['actual_vs_predicted']['actual']) == 8
    assert len(dashboard['charts']['residuals']['residuals']) == 8
    assert dashboard['charts']['confusion_matrix']['values'] is None
