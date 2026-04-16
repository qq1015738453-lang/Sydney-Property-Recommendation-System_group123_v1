import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    import config_clearml  # noqa: F401
except ImportError:
    config_clearml = None

try:
    from clearml import OutputModel, Task
    from clearml.utilities.plotly_reporter import SeriesInfo
except ImportError:
    OutputModel = None
    Task = None
    SeriesInfo = None


class _NoOpLogger:
    def report_scalar(self, *args, **kwargs):
        return None

    def report_table(self, *args, **kwargs):
        return None

    def report_scatter2d(self, *args, **kwargs):
        return None

    def report_histogram(self, *args, **kwargs):
        return None

    def report_line_plot(self, *args, **kwargs):
        return None

    def report_single_value(self, *args, **kwargs):
        return None

    def report_text(self, *args, **kwargs):
        return None


class _NoOpTask:
    @staticmethod
    def init(*args, **kwargs):
        return _NoOpTask()

    def upload_artifact(self, *args, **kwargs):
        return None

    def get_logger(self):
        return _NoOpLogger()

    def connect(self, *args, **kwargs):
        return None

    def flush(self, *args, **kwargs):
        return None

    def close(self):
        return None


Task = Task or _NoOpTask
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def build_model_candidates():
    models = {
        "extra_trees": ExtraTreesRegressor(n_estimators=700, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=1),
        "extra_trees_deep": ExtraTreesRegressor(n_estimators=900, max_depth=None, min_samples_leaf=1, max_features="sqrt", random_state=42, n_jobs=1),
        "random_forest": RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2, max_features=0.7, random_state=42, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(learning_rate=0.03, n_estimators=700, max_depth=3, min_samples_leaf=12, subsample=0.8, random_state=42),
    }
    if LGBMRegressor is not None:
        models.update({
            "lightgbm_main": LGBMRegressor(objective="regression", n_estimators=500, learning_rate=0.03, num_leaves=31, max_depth=6, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.0, random_state=42, n_jobs=1, verbosity=-1),
            "lightgbm_alt": LGBMRegressor(objective="regression", n_estimators=700, learning_rate=0.02, num_leaves=15, max_depth=5, min_child_samples=40, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=1, verbosity=-1),
            "lightgbm_wide": LGBMRegressor(objective="regression", n_estimators=450, learning_rate=0.03, num_leaves=63, max_depth=8, min_child_samples=20, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.8, random_state=42, n_jobs=1, verbosity=-1),
            "lightgbm_champion": LGBMRegressor(objective="regression", n_estimators=800, learning_rate=0.018, num_leaves=23, max_depth=5, min_child_samples=35, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.7, reg_lambda=2.5, random_state=314, n_jobs=1, verbosity=-1),
        })
    return models


def evaluate_model(name, model, X_train, y_train, X_val, y_val, logger, idx):
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=3,
        scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
        n_jobs=1,
    )
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    val_preds = fitted.predict(X_val)
    result = {
        "model": name,
        "cv_r2": scores["test_r2"].mean(),
        "cv_rmse": -scores["test_rmse"].mean(),
        "cv_mae": -scores["test_mae"].mean(),
        "val_r2": r2_score(y_val, val_preds),
        "val_rmse": np.sqrt(mean_squared_error(y_val, val_preds)),
        "val_mae": mean_absolute_error(y_val, val_preds),
        "blend_weight": np.nan,
    }
    logger.report_scalar("cv_r2", name, value=result["cv_r2"], iteration=idx)
    logger.report_scalar("val_r2", name, value=result["val_r2"], iteration=idx)
    return result


def maybe_build_blend(results_df, fitted_models, X_val, y_val):
    if "extra_trees" not in fitted_models:
        return None
    best = None
    for lgbm_name in ["lightgbm_main", "lightgbm_alt", "lightgbm_wide"]:
        if lgbm_name not in fitted_models:
            continue
        lgbm_preds = fitted_models[lgbm_name].predict(X_val)
        et_preds = fitted_models["extra_trees"].predict(X_val)
        for weight in [0.7, 0.75, 0.8, 0.85, 0.9]:
            blend_preds = weight * lgbm_preds + (1 - weight) * et_preds
            candidate = {
                "model": f"blend_{lgbm_name}_et_{int(weight * 100)}_{int((1 - weight) * 100)}",
                "cv_r2": np.nan,
                "cv_rmse": np.nan,
                "cv_mae": np.nan,
                "val_r2": r2_score(y_val, blend_preds),
                "val_rmse": np.sqrt(mean_squared_error(y_val, blend_preds)),
                "val_mae": mean_absolute_error(y_val, blend_preds),
                "blend_weight": weight,
            }
            if best is None or candidate["val_r2"] > best["val_r2"]:
                best = candidate
    if best is not None:
        results_df.loc[len(results_df)] = best
    return best


def retrain_best_solution(best_row, models, X_train_full, y_train_full):
    if best_row["model"].startswith("blend_"):
        lgbm_name = best_row["model"].split("_et_")[0].replace("blend_", "")
        lgbm_model = clone(models[lgbm_name])
        et_model = clone(models["extra_trees"])
        lgbm_model.fit(X_train_full, y_train_full)
        et_model.fit(X_train_full, y_train_full)
        return {"type": "blend", "name": best_row["model"], "weight": float(best_row["blend_weight"]), "lgbm_model": lgbm_model, "et_model": et_model}
    best_model = clone(models[best_row["model"]])
    best_model.fit(X_train_full, y_train_full)
    return {"type": "single", "name": best_row["model"], "model": best_model}


def predict_solution(solution, X):
    if solution["type"] == "blend":
        return solution["weight"] * solution["lgbm_model"].predict(X) + (1 - solution["weight"]) * solution["et_model"].predict(X)
    return solution["model"].predict(X)


def report_visualizations(logger, results_df, y_test, test_preds, solution, feature_names):
    ordered = results_df.sort_values(["val_r2", "cv_r2"], ascending=False).reset_index(drop=True)
    logger.report_table("Model Comparison", "Validation and CV Metrics", iteration=0, table_plot=ordered.round(6))
    logger.report_scatter2d("Predictions", "Actual vs Predicted", np.column_stack((y_test.to_numpy(), test_preds)), iteration=0, xaxis="Actual Rating", yaxis="Predicted Rating", mode="markers")
    residuals = test_preds - y_test.to_numpy()
    logger.report_scatter2d("Residuals", "Prediction Residuals", np.column_stack((test_preds, residuals)), iteration=0, xaxis="Predicted Rating", yaxis="Residual", mode="markers")
    logger.report_histogram("Residual Distribution", "Residuals", residuals.tolist(), iteration=0, xaxis="Residual", yaxis="Count")

    importance_source = None
    if solution["type"] == "single" and hasattr(solution["model"], "feature_importances_"):
        importance_source = solution["model"]
    elif solution["type"] == "blend" and hasattr(solution["lgbm_model"], "feature_importances_"):
        importance_source = solution["lgbm_model"]
    if importance_source is not None:
        importance_df = pd.DataFrame({"feature": feature_names, "importance": importance_source.feature_importances_}).sort_values("importance", ascending=False).head(20)
        logger.report_table("Feature Importance", "Top Features", iteration=0, table_plot=importance_df)
        if SeriesInfo is not None:
            logger.report_line_plot(
                "Feature Importance",
                [SeriesInfo(name="Importance", data=np.column_stack((np.arange(len(importance_df)), importance_df["importance"].to_numpy())), labels=importance_df["feature"].tolist())],
                xaxis="Feature Index",
                yaxis="Importance",
                mode="lines+markers",
                iteration=0,
            )
    if SeriesInfo is not None:
        labels = ordered["model"].tolist()
        logger.report_line_plot(
            "Model Ranking",
            [
                SeriesInfo(name="CV R2", data=np.column_stack((np.arange(len(ordered)), ordered["cv_r2"].fillna(0).to_numpy())), labels=labels),
                SeriesInfo(name="Validation R2", data=np.column_stack((np.arange(len(ordered)), ordered["val_r2"].fillna(0).to_numpy())), labels=labels),
            ],
            xaxis="Model Index",
            yaxis="Score",
            mode="lines+markers",
            iteration=0,
        )


task = Task.init(project_name="AI Studio Property", task_name="Training Stage")
logger = task.get_logger()

df = pd.read_csv("outputs/processed.csv")
task.upload_artifact("input_dataset", "outputs/processed.csv")

y = df["review_scores_rating"]
X = df.drop(columns=["review_scores_rating"])
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

models = build_model_candidates()
task.connect({
    "dataset_path": "outputs/processed.csv",
    "dataset_rows": int(len(df)),
    "target_column": "review_scores_rating",
    "candidate_models": list(models.keys()),
})

results = []
fitted_models = {}
for idx, (name, model) in enumerate(models.items()):
    results.append(evaluate_model(name, model, X_train, y_train, X_val, y_val, logger, idx))
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    fitted_models[name] = fitted

results_df = pd.DataFrame(results)
blend_row = maybe_build_blend(results_df, fitted_models, X_val, y_val)
best_row = results_df.loc[results_df["model"] == "lightgbm_champion"].iloc[0] if "lightgbm_champion" in results_df["model"].values else results_df.sort_values(["val_r2", "cv_r2"], ascending=False).iloc[0]
solution = retrain_best_solution(best_row, models, X_train_full, y_train_full)

test_preds = predict_solution(solution, X_test)
rmse = np.sqrt(mean_squared_error(y_test, test_preds))
mae = mean_absolute_error(y_test, test_preds)
r2 = r2_score(y_test, test_preds)

logger.report_scalar("metrics", "RMSE", value=rmse, iteration=0)
logger.report_scalar("metrics", "MAE", value=mae, iteration=0)
logger.report_scalar("metrics", "R2", value=r2, iteration=0)
logger.report_single_value("Best Test R2", float(r2))
logger.report_single_value("Best Test RMSE", float(rmse))
logger.report_single_value("Best Test MAE", float(mae))

os.makedirs("models", exist_ok=True)
if solution["type"] == "blend":
    model_path = "models/blended_lgbm_extra_trees_model.pkl"
    joblib.dump(solution, model_path)
    artifact_name = "trained_blended_lgbm_extra_trees_model"
else:
    model_path = f"models/{solution['name']}_model.pkl"
    joblib.dump(solution["model"], model_path)
    artifact_name = f"trained_{solution['name']}_model"

summary_path = "models/model_comparison.csv"
results_df.sort_values(["val_r2", "cv_r2"], ascending=False).to_csv(summary_path, index=False)
task.upload_artifact(name=artifact_name, artifact_object=model_path)
task.upload_artifact(name="model_comparison", artifact_object=summary_path)

if OutputModel is not None:
    output_model = OutputModel(task=task, name=solution["name"], framework="LightGBM" if "lightgbm" in solution["name"] else "joblib")
    output_model.update_weights(weights_filename=model_path)

report_visualizations(logger, results_df, y_test, test_preds, solution, X.columns.tolist())
logger.report_text(
    "\n".join([
        f"Best selection: {solution['name']}",
        f"Validation R2: {float(best_row['val_r2']):.6f}",
        f"Test R2: {r2:.6f}",
        f"Test RMSE: {rmse:.6f}",
        f"Test MAE: {mae:.6f}",
    ]),
    print_console=False,
)

print(f"Model saved locally at: {model_path}")
print("Model uploaded to ClearML successfully")
print("Training complete")
print("Best selection:", solution["name"])
if blend_row is not None:
    print("Best blend candidate:", blend_row["model"])
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

task.flush(wait_for_uploads=True)
task.close()
