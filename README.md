# Credit Card Fraud Detection

This project demonstrates a complete machine learning pipeline for detecting fraudulent credit card transactions. It is designed for internship or academic submission and includes both a Jupyter notebook for step-by-step exploration and a Python script for reproducible execution.

## Project Structure
- `data/` : Contains the dataset (`creditcard.csv` from Kaggle)
- `notebooks/` : Jupyter notebook for data exploration and modeling
- `main.py` : Main script to run the pipeline
- `requirements.txt` : Python dependencies

## How to Run
1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Download the dataset:**
   - Get `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Place it in the `data/` folder.
3. **Run the notebook:**
   - Open `notebooks/credit_card_fraud_detection.ipynb` in Jupyter or VS Code.
   - The notebook uses a 10% random sample of the data for faster development and demonstration. For final results, remove or adjust the sampling line.
4. **Run the script:**
   ```powershell
   python main.py
   ```
   - The script also uses a 10% sample for speed. Adjust as needed for full dataset runs.

## Features
- Data loading, exploration, and visualization
- Preprocessing: scaling, train-test split, and class balancing with SMOTE
- Model training: Logistic Regression and Random Forest
- Evaluation: Classification report, ROC-AUC, confusion matrix, and ROC curve plots
- Well-commented code and clear outputs

## Notes for Reviewers
- The project uses a 10% sample of the data for demonstration and speed. For production or final evaluation, use the full dataset by removing the sampling line in both the notebook and script.
- All code is ready to run and well-documented for internship review.

---

**Author:** Himanshu Karan  
**Date:** May 31, 2025
