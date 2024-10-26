# Calories Burned Prediction

This project aims to predict the number of calories burned during exercise using various machine learning models. It uses data from two CSV files: 'calories.csv' and 'exercise.csv'.

## Table of Contents
1. [Dependencies](#dependencies)
2. [Data Preparation](#data-preparation)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Feature Engineering](#feature-engineering)
5. [Model Training and Evaluation](#model-training-and-evaluation)

## Dependencies

This project requires the following Python libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install these dependencies using pip:
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

## Data Preparation

The project starts by loading two datasets:
- 'calories.csv': Contains information about calories burned
- 'exercise.csv': Contains exercise-related features

These datasets are merged, and duplicate columns are removed.

## Exploratory Data Analysis

Several visualizations are created to understand the data:
1. A scatter plot of Height vs Weight
2. Scatter plots of Age, Height, Weight, and Duration vs Calories
3. Distribution plots for numerical features

## Feature Engineering

- The 'Gender' column is encoded (male: 0, female: 1)
- A correlation heatmap is created to identify highly correlated features
- 'Weight' and 'Duration' columns are removed to avoid multicollinearity

## Model Training and Evaluation

The data is split into training and validation sets (90% training, 10% validation).

Features are standardized using StandardScaler.

Five different models are trained and evaluated:
1. Linear Regression
2. XGBoost Regressor
3. Lasso Regression
4. Random Forest Regressor
5. Ridge Regression

For each model, the Mean Absolute Error (MAE) is calculated for both training and validation sets.

## Usage

To run this project:
1. Ensure all dependencies are installed
2. Place 'calories.csv' and 'exercise.csv' in the same directory as the script
3. Run the Python script

The script will output visualizations and model performance metrics.
