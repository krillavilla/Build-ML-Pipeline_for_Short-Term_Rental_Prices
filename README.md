# NYC Airbnb Price Prediction Pipeline

## Project Overview
I developed an end-to-end machine learning pipeline that predicts Airbnb rental prices in New York City. This project showcases my expertise in MLOps, demonstrating my ability to build production-ready machine learning systems using industry-standard tools and best practices.

Key achievements and technical highlights:
- Engineered a robust ML pipeline using MLflow for workflow orchestration and experiment tracking
- Implemented comprehensive data validation and cleaning processes to ensure data quality
- Developed an advanced feature engineering system, including TF-IDF text processing for listing descriptions
- Built and optimized a Random Forest model achieving strong predictive performance
- Utilized Weights & Biases (W&B) for experiment tracking, model versioning, and artifact management
- Applied software engineering best practices including modular design, version control, and continuous integration

Technologies used: Python, MLflow, Weights & Biases, scikit-learn, pandas, hydra

## Links
- Weights & Biases Project: https://wandb.ai/krillavilla-arizona-state-university/nyc_airbnb?nw=nwuserkrillavilla
- GitHub Repository: https://github.com/krillavilla/Build-ML-Pipeline_for_Short-Term_Rental_Prices

## Project Structure
- `main.py`: Main pipeline orchestration
- `config.yaml`: Configuration file for pipeline parameters
- `src/`: Source code for individual pipeline components
  - `basic_cleaning/`: Data cleaning component
  - `data_check/`: Data validation component
  - `train_random_forest/`: Model training component

## Running the Pipeline
```bash
python main.py
```
