# Palantir-Pipline-DEV


## ML Model

## Overview
**AI-Pipeline-DEV** is a machine learning pipeline developed as part of the AIM-AHEAD initiative. This repository focuses on the development, testing, and deployment of AI/ML models for analyzing healthcare data, with an emphasis on causal inference and predictive modeling in maternal and fetal health outcomes.

## Objectives
- Develop and evaluate ML models using structured clinical datasets
- Apply causal inference techniques to uncover meaningful relationships
- Facilitate reproducible, scalable workflows for healthcare research
- Leverage data from the N3C Data Enclave for respiratory disease and pregnancy studies

## Features
- Data preprocessing and standardization
- Exploratory Data Analysis (EDA) tools
- Feature engineering pipeline
- ML model training (Random Forest, XGBoost, Logistic Regression, etc.)
- Causal inference (e.g., propensity score matching, inverse probability weighting)
- Evaluation metrics and visualizations
- Deployment-ready modular design

## ML Model
We currently support the following machine learning models:
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machines (SVM)
- Neural Networks (planned)
- Deep Neural Network

All models are integrated into a modular training pipeline and support hyperparameter tuning, cross-validation, and interpretability tools (e.g., SHAP values, feature importance).

## Folder Structure
# Platform

This pipeline is developed and run on **Palantir Foundry**, which provides scalable infrastructure for secure data integration, transformation, and analysis. Foundry enables seamless collaboration and reproducibility across the research team.
### Repository Layout

### Key Script: `palantir_pipeline_facttable_dev.py`
- Filters the patient fact table to isolate **female delivery patients** using Palantir concept sets.
- Applies transformations using Spark and Palantir’s `@transform_df` decorators.
- Outputs a clean delivery-only cohort for downstream analysis.
