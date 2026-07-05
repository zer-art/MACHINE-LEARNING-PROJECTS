# Tweet Sentiment Predictor

Small Streamlit app and model bundle for predicting tweet sentiment using a logistic regression pipeline.

**Contents**
- `main.py` — Streamlit entrypoint that loads the model and starts the app.
- `ui/app.py` — Streamlit UI renderer and helpers.
- `models/` — Serialized model and metadata (`logistic_sentiment_pipeline.joblib`, `model_metadata.json`).
- `notebooks/` — Training notebooks; run `notebooks/train.ipynb` to retrain and export the model.

**Prerequisites**
- Python 3.11 (project uses `pixi.toml` pinned to 3.11.x)
- Pixi (project environment manager) — dependencies are declared in `pixi.toml`
- Git (optional)

**Dependencies**
All runtime dependencies are listed in `pixi.toml` (e.g. `streamlit`, `pandas`, `scikit-learn`, `joblib`). Use Pixi to create a reproducible environment.

Installation (using Pixi)

1. Install Pixi following the Pixi documentation for your platform.

2. From the project root, create the environment and install dependencies declared in `pixi.toml`:

```bash
pixi install
```

3a. Run a command directly inside the Pixi environment:

```bash
pixi run streamlit run main.py
```

3b. Or open an interactive shell inside the Pixi environment and run commands there:

```bash
pixi shell
streamlit run main.py
```

Alternative: virtual environment (fallback)

If you prefer not to use Pixi, create a virtual environment and install the primary dependencies manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install streamlit pandas scikit-learn joblib
streamlit run main.py
```

Run the app (development)

- Use `pixi run streamlit run main.py` from the project root (recommended when using Pixi).
- The Streamlit server will open a browser window or print a local URL to visit.

Notes on running

- `main.py` expects a trained model at `models/logistic_sentiment_pipeline.joblib`. If that file is missing, the app will show an error and suggest running the training notebook at `notebooks/train.ipynb`.
- To retrain or update the model, open `notebooks/train.ipynb` in Jupyter / VS Code and run the cells; export the trained pipeline to `models/logistic_sentiment_pipeline.joblib`.

Project structure

- `main.py` — Streamlit entrypoint that loads the model and calls `render_app()`.
- `ui/app.py` — UI layout and prediction UI; uses `pandas` and `streamlit`.
- `models/` — Contains exported model and metadata used by the app.
- `data/` — (Optional) CSV datasets used for training (`data/train.csv`).
- `notebooks/` — Training and evaluation notebooks (`train.ipynb`, `test_report.ipynb`).

Troubleshooting

- If the web UI does not start, ensure Pixi dependencies were installed with `pixi install` and run the app using `pixi run streamlit run main.py`.
- If `models/logistic_sentiment_pipeline.joblib` is missing, run the training notebook or obtain the model file from your experiment outputs.
- If you see errors about missing packages, run `pixi install` again or open an environment shell with `pixi shell` and install packages as needed.

Optional: export a `requirements.txt`

To capture the environment from within Pixi:

```bash
pixi run pip freeze > requirements.txt
```

Contact

For questions about the project, reach out to the repository owner or the author of the training notebooks.
