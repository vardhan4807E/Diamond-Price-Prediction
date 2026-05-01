# 💎 Diamond Price Prediction using KNN and Streamlit

This project predicts diamond prices using a K-Nearest Neighbors Regression model.

## Objective
To build an end-to-end machine learning application that predicts diamond price based on diamond features.

## Dataset
- diamonds.csv
- Target variable: price

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Pickle
- Streamlit

## ML Pipeline
1. Load dataset
2. Split input and target
3. Train-test split 75:25
4. Encode categorical columns
5. Scale numerical columns
6. Train KNN Regressor
7. Evaluate model using MAE, RMSE, and R² Score
8. Save model using Pickle
9. Deploy using Streamlit

## Run Locally

```bash
streamlit run app.py
