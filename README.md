# Sydney Property Recommendation System

A machine learning pipeline for predicting Airbnb-style property review ratings for Sydney listings.

This public package is prepared for GitHub upload and keeps secrets and local runtime outputs out of version control.

## Features

- Feature engineering from treated listing data
- Metadata merge from a secondary listings table
- Tree-based regression model comparison
- Optional ClearML experiment tracking
- Model export and evaluation reporting

## Repository Structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── .env.example
├── config_clearml.py
├── preprocess.py
├── train.py
├── run_pipeline.py
└── data/
    └── README.md
```

## Data

The datasets are not included in this repository.

Before running the project, create a `data/` directory and place:

- `data/half_treated.csv`
- `data/listings.csv`

See `data/README.md` for the expected layout.

## Installation

```bash
pip install -r requirements.txt
```

## Run

Run the full pipeline:

```bash
python run_pipeline.py
```

Or run each stage manually:

```bash
python preprocess.py
python train.py
```

## ClearML Setup

ClearML is optional.

You can either run `clearml-init` or use environment variables from `.env.example`:

- `CLEARML_API_HOST`
- `CLEARML_WEB_HOST`
- `CLEARML_FILES_HOST`
- `CLEARML_API_ACCESS_KEY`
- `CLEARML_API_SECRET_KEY`

`config_clearml.py` reads those values automatically when they are present.

## Outputs

After a successful run, the project writes:

- `outputs/processed.csv`
- `models/model_comparison.csv`
- a trained model file such as `models/lightgbm_champion_model.pkl`

If ClearML is configured, the training task can also log:

- scalar metrics such as `R2`, `RMSE`, and `MAE`
- model comparison tables
- prediction and residual plots
- feature importance summaries
- registered model artifacts

## Notes

- Do not upload real dataset files unless redistribution is allowed.
- Do not hardcode credentials in source files.
- Keep `data/`, `models/`, and `outputs/` out of Git unless you intentionally want them published.
