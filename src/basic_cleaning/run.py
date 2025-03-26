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
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    # Read the data
    df = pd.read_csv(artifact_path)

    # Convert price to float
    logger.info("Converting price to float")
    df['price'] = df['price'].astype(float)

    # Drop outliers
    logger.info("Cleaning price outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Add NYC geographical boundaries filtering
    logger.info("Filtering data for NYC boundaries")
    
    # NYC coordinates boundaries (approximate)
    nyc_bounds = {
        'lat': {'min': 40.4774, 'max': 40.9176},
        'long': {'min': -74.2591, 'max': -73.7004}
    }
    
    idx = (
        df['latitude'].between(nyc_bounds['lat']['min'], nyc_bounds['lat']['max']) &
        df['longitude'].between(nyc_bounds['long']['min'], nyc_bounds['long']['max'])
    )
    
    df = df[idx].copy()
    logger.info(f"Removed {(~idx).sum()} records outside NYC boundaries")

    # Save the cleaned data
    logger.info("Saving cleaned data")
    df.to_csv("clean_sample.csv", index=False)

    logger.info("Creating cleaned artifact")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    # Log metrics about the cleaning 
    run.log({
        "n_records_input": len(pd.read_csv(artifact_path)),
        "n_records_output": len(df),
        "n_outside_nyc": (~idx).sum()
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the raw data")
    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact name",
        required=True
    )
    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output artifact name",
        required=True
    )
    parser.add_argument(
        "--output_type", 
        type=str,
        help="Output artifact type",
        required=True
    )
    parser.add_argument(
        "--output_description", 
        type=str,
        help="Output artifact description",
        required=True
    )
    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price cutoff",
        required=True
    )
    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price cutoff",
        required=True
    )

    args = parser.parse_args()
    go(args)