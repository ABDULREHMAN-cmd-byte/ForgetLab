# ForgetLab

ForgetLab is a machine unlearning web application built with Flask and scikit-learn. It demonstrates how to train a baseline model, apply exact retraining after forgetting selected records, and compare that result against an experimental approximate unlearning workflow.

## Features

- Upload and validate CSV datasets
- Train logistic regression and random forest models
- Support the Adult Census Income dataset
- Exact retraining baseline for record removal
- Approximate unlearning experiment using SGDClassifier
- Evaluate metrics and save experiment summaries
- Dark premium dashboard for monitoring workflows

## Quick start

1. Create or use the project virtual environment.
2. Install dependencies:
   
   .venv\Scripts\python.exe -m pip install -r requirements.txt

3. Run the app:

   .venv\Scripts\python.exe app.py

4. Open http://127.0.0.1:5000 in a browser.

## Deploy on Render

The root `render.yaml` configures the web service. Render will install
`requirements.txt` and start the app with:

```text
gunicorn wsgi:app --bind 0.0.0.0:$PORT
```

The Blueprint attaches a 1 GB persistent disk at `/var/data`; the database,
uploads, trained models, and generated results are stored there. Persistent
disks require a paid Render web service. The included `1c-2g` plan is selected
for this Blueprint; review Render's current pricing before creating it. Free
Render services use ephemeral storage and cannot attach a persistent disk.

To deploy, push the project to a Git provider, create a Render Blueprint from
the repository, and review the service plan and disk before applying it.

## Project structure

- app.py: Flask entry point
- config.py: settings and file locations
- database.py: database initialization
- models_db.py: SQLAlchemy metadata models
- machine_learning/: training, dataset, evaluation, unlearning, visualization modules
- templates/: HTML pages
- static/: CSS and JS assets
- tests/: pytest test suite

## Notes

This project emphasizes scientifically honest unlearning behavior. Exact retraining is a real baseline method, while the approximate unlearning workflow is framed as an experiment rather than a guaranteed removal proof.
