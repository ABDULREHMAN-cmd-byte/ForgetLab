# Testing Report

The project uses pytest to validate dataset handling, training, unlearning, and route startup behavior.

## How to run tests

```powershell
cd "d:\Forget Lab"
.\.venv\Scripts\python.exe -m pytest -q
```

## Intended coverage

- Dataset validation
- CSV upload processing
- Model training
- Exact retraining
- Approximate unlearning
- App startup and health endpoint
