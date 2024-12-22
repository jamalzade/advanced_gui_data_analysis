# advanced_gui_data_analysis

# Advanced Data Analysis Tool

## Overview

The Advanced Data Analysis Tool is an interactive Python application designed for data scientists and analysts to quickly load datasets, preprocess data, and apply a variety of machine learning models. Built with an intuitive Tkinter-based GUI, it supports Regression, Classification, and Clustering tasks.

## Features

- **Data Loading**: Supports CSV and Excel file formats for input.
- **Regression Models**:
  - Random Forest Regressor
  - Linear Regression
  - Decision Tree Regressor
- **Classification Models**:
  - Random Forest Classifier
  - Logistic Regression
  - Decision Tree Classifier
- **Clustering Techniques**:
  - K-Means Clustering
  - Principal Component Analysis (PCA)
- **Preprocessing**:
  - Handles missing values using mean imputation.
  - Normalizes features using StandardScaler.
- **Visualization**:
  - Regression performance plots.
  - Classification heatmaps.
  - PCA and Clustering scatter plots.
- **Cross-validation**: Provides cross-validation scores for regression and classification models.

## Requirements

- Python 3.7 or above
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - tkinter (comes pre-installed with Python)

## How to Use

1. **Load Dataset**:
   - Click the "Load Dataset" button and select a CSV or Excel file.
   - The dataset summary is displayed, including shape, missing values, and available columns.
2. **Select Model Type**:
   - Choose between Regression, Classification, or Clustering.
3. **Select Model**:
   - Pick a model from the dropdown menu corresponding to the selected task.
4. **Analyze and Build Model**:
   - Click the "Analyze and Build Model" button to train the model or perform clustering.
   - View the results in the Results section or via visualizations.

## Example Workflow

1. Load a dataset containing numeric features and a target variable.
2. Choose **Regression** and select **Random Forest**.
3. Click "Analyze and Build Model" to evaluate the model using metrics like MSE, RMSE, and R-squared.
4. Switch to **Clustering**, select **PCA**, and visualize the data's principal components.

## File Structure

- `main.py`: The primary script to launch the application.
- `data_analysis.log`: Logs detailed runtime information for debugging and monitoring.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Enjoy analyzing your data with ease and efficiency!
"""
