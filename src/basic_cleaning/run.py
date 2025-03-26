#!/usr/bin/env python
"""
This script cleans the input data
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    try:
        run = wandb.init(
            project="nyc_airbnb",
            entity="build-ml-pipeline-for-short-term-rental-prices",
            job_type="basic_cleaning",
            resume=True
        )
        run.config.update(args)
    except Exception as e:
        logger.error(f"Error during W&B initialization: {e}")
        raise e

    # Download input artifact
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    # Read the data
    logger.info("Reading data from artifact")
    df = pd.read_csv(artifact_path)

    # Convert price to float
    df['price'] = df['price'].astype(float)

    # Drop outliers
    logger.info("Cleaning data")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Add geographical boundary filtering
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save the cleaned data
    logger.info("Saving cleaned data")
    df.to_csv("clean_sample.csv", index=False)

    # Upload the cleaned data as an artifact
    logger.info("Creating artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    logger.info("Logging artifact")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price for cleaning",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price for cleaning",
        required=True
    )

    args = parser.parse_args()

    go(args)