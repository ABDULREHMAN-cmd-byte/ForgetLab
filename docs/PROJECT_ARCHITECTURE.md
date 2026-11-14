# Project Architecture

ForgetLab follows a simple separation of concerns pattern. The backend owns the Flask app, the database layer stores metadata, and the `machine_learning` package contains the ML logic.

## Main layers

- Application layer: Flask routes and views
- Data layer: SQLite + SQLAlchemy models
- ML layer: preprocessing, training, unlearning, and evaluation
- Presentation layer: Bootstrap dashboard templates
- Reporting layer: JSON and summary generation

## Why this structure matters

A clear module boundary makes the application easier to debug, extend, and test. It also separates model logic from interface logic so the same training pipeline can be reused by a web app or a notebook.
