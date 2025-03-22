#!/usr/bin/env python
"""
This script trains a Random Forest model on the cleaned data
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def process_features(df, tfidf_max_features=5):
    """
    Process features including text vectorization
    Returns:
        X: feature matrix
        feature_names: list of feature names
    """
    logger.info("Processing features")
    
    # Text feature processing
    tfidf = TfidfVectorizer(max_features=tfidf_max_features)
    name_tfidf = tfidf.fit_transform(df['name'].fillna('')).toarray()
    
    # Create feature matrix
    features = ['room_type', 'neighbourhood_group']
    X_cat = pd.get_dummies(df[features], drop_first=True)
    
    # Numerical features
    numerical_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews']
    X_numerical = df[numerical_features]
    
    # Combine all features
    X = np.concatenate([X_numerical, X_cat, name_tfidf], axis=1)
    
    # Create feature names list
    feature_names = (
        list(X_numerical.columns) + 
        list(X_cat.columns) + 
        [f'tfidf_{i}' for i in range(tfidf_max_features)]
    )
    
    return X, feature_names

def train_random_forest(args):
    """
    Train the random forest model
    """
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Download and read training data
    logger.info("Downloading training data")
    local_path = run.use_artifact(args.training_data).file()
    df = pd.read_csv(local_path)

    # Process features
    X, feature_names = process_features(df, args.max_tfidf_features)
    y = df['price'].values

    # Train/validation split
    logger.info("Splitting training and validation data")
    train_idx, val_idx = train_test_split(
        np.arange(len(X)),
        test_size=args.val_size,
        random_state=args.random_seed
    )
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Train model
    logger.info("Training model")
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_seed,
        n_jobs=args.n_jobs
    )
    
    rf.fit(X_train, y_train)

    # Create a pipeline that includes feature processing
    from sklearn.pipeline import Pipeline
    
    # Create feature processor class
    class FeatureProcessor:
        def __init__(self, tfidf_max_features=5):
            self.tfidf = TfidfVectorizer(max_features=tfidf_max_features)
            self.tfidf_max_features = tfidf_max_features
            
        def fit(self, X, y=None):
            # Fit TF-IDF on the 'name' column
            self.tfidf.fit(X['name'].fillna(''))
            return self
            
        def transform(self, X):
            # Text feature processing
            name_tfidf = self.tfidf.transform(X['name'].fillna('')).toarray()
            
            # Create feature matrix
            features = ['room_type', 'neighbourhood_group']
            X_cat = pd.get_dummies(X[features], drop_first=True)
            
            # Numerical features
            numerical_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews']
            X_numerical = X[numerical_features]
            
            # Combine all features
            X_processed = np.concatenate([X_numerical, X_cat, name_tfidf], axis=1)
            return X_processed
    
    # Create and fit the pipeline
    pipeline = Pipeline([
        ('feature_processor', FeatureProcessor(tfidf_max_features=args.max_tfidf_features)),
        ('regressor', rf)
    ])
    
    pipeline.fit(df, y)  # Fit the entire pipeline

    # Evaluate
    logger.info("Evaluating model")
    val_data = df.iloc[val_idx]  # Get validation data using saved indices
    y_pred = pipeline.predict(val_data)  # Predict on validation data
    
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)

    # Log metrics
    run.summary['mae'] = mae
    run.summary['mse'] = mse
    run.summary['rmse'] = rmse
    run.summary['r2'] = r2

    # Plot feature importance (using the RF model's feature importances)
    feature_importance = pd.DataFrame(
        rf.feature_importances_,
        index=feature_names,
        columns=['importance']
    ).sort_values('importance', ascending=False)

    fig, ax = plt.subplots()
    feature_importance.head(10).plot(kind='barh', ax=ax)
    ax.set_title("Top 10 Feature Importance")
    plt.tight_layout()
    run.log({"feature_importance": wandb.Image(fig)})
    plt.close()

    # Export pipeline
    logger.info("Exporting model")
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    mlflow.sklearn.save_model(pipeline, "random_forest_dir")

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type="model_export",
        description="Random Forest pipeline export"
    )
    artifact.add_dir("random_forest_dir")
    run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest model")
    parser.add_argument(
        "--training_data",
        type=str,
        help="Training data artifact"
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of validation split"
    )
    parser.add_argument(
        "--max_tfidf_features",
        type=int,
        help="Maximum number of words to consider for TFIDF"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        help="Number of trees in the forest",
        default=100
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        help="Maximum tree depth",
        default=10
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        help="Minimum samples required to split",
        default=2
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        help="Minimum samples required at leaf node"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Number of parallel jobs"
    )

    args = parser.parse_args()
    train_random_forest(args)
