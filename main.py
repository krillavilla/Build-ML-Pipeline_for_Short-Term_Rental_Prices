import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

# Only set project and entity - API key should be set via environment variable or wandb login
os.environ["WANDB_PROJECT"] = "nyc_airbnb"
os.environ["WANDB_ENTITY"] = "build-ml-pipeline-for-short-term-rental-prices"

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    "test_regression_model"
]

def get_best_model(api):
    """Get the best performing model from W&B"""
    try:
        # Get all runs from the project
        project_path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
        
        # Use a more specific query
        runs = api.runs(
            path=project_path,
            filters={
                "config.job_type": "train_random_forest",
                "state": "finished"
            }
        )
        
        # Find the run with the lowest MAE
        best_mae = float('inf')
        best_run = None
        
        for run in runs:
            try:
                mae = run.summary.get('mae', float('inf'))
                if mae < best_mae:
                    best_mae = mae
                    best_run = run
            except Exception as e:
                print(f"Error processing run {run.id}: {e}")
                continue
        
        return best_run
    except Exception as e:
        print(f"Error fetching best model: {e}")
        return None

@hydra.main(config_path=".", config_name="config", version_base=None)
def go(config: DictConfig):
    # Initialize W&B first
    run = wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        job_type="pipeline",
    )
    
    # These will override the previous settings if specified in config
    if "project_name" in config["main"]:
        os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    if "experiment_name" in config["main"]:
        os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # After pipeline completion, set up tags
    api = wandb.Api()
    
    # Tag the latest clean_sample.csv as reference
    if "basic_cleaning" in active_steps:
        try:
            # Get the latest version of clean_sample.csv
            artifact = api.artifact(
                f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/clean_sample.csv:latest"
            )
            # Add the reference tag
            artifact.link('reference')
            print("Tagged clean_sample.csv:latest as reference")
        except Exception as e:
            print(f"Error tagging clean_sample.csv: {e}")
    
    # Tag best model as prod
    if "train_random_forest" in active_steps:
        best_run = get_best_model(api)
        if best_run is not None:
            try:
                # Get the model artifact from the best run
                for artifact in best_run.logged_artifacts():
                    if artifact.type == "model_export":
                        artifact.link('prod')
                        print(f"Tagged best model from run {best_run.name} as prod")
                        break
            except Exception as e:
                print(f"Error tagging best model: {e}")

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "get_data"),
                "main",
                version=None,  # Remove version since we're using local path
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "train_val_test_split"),
                "main",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                },
            )

        if "train_random_forest" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                "main",
                parameters={
                    "training_data": "trainval_data.csv:latest",
                    "output_artifact": "random_forest_export",
                    "random_seed": config["modeling"]["random_seed"],
                    "val_size": config["modeling"]["val_size"],
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "n_estimators": config["modeling"]["random_forest"]["n_estimators"],
                    "max_depth": config["modeling"]["random_forest"]["max_depth"],
                    "min_samples_split": config["modeling"]["random_forest"]["min_samples_split"],
                    "min_samples_leaf": config["modeling"]["random_forest"]["min_samples_leaf"],
                    "n_jobs": config["modeling"]["random_forest"]["n_jobs"]
                },
            )

        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "test_regression_model"),
                "main",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest"
                },
            )


if __name__ == "__main__":
    go()
