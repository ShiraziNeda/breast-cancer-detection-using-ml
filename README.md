﻿# breast_cancer_detection_project
Title:
Breast Cancer Detection using Machine Learning

Overview:
This project leverages the breast_cancer dataset from sklearn to detect breast cancer. Multiple machine learning models are trained, evaluated, and compared to identify the best-performing model.

Key Features:
Data preprocessing (splitting, normalization).
Training and testing of various classification models:
Naive Bayes
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
Support Vector Machine (SVM)
Logistic Regression
Artificial Neural Network (ANN)
Performance metrics: Accuracy, Precision, Recall.
Visual comparison of model performances.
Technologies Used:
Python
Libraries: Scikit-learn, Matplotlib, NumPy
Setup:
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/breast-cancer-detection.git
Install the required libraries:
bash
Copy code
pip install -r requirements.txt
Run the Jupyter Notebook:
bash
Copy code
jupyter notebook breast_cancer_detection.ipynb

Results:
Comparison of accuracy, precision, and recall for all models:

Model	Accuracy (Train)	Accuracy (Test)	Precision	Recall
Naive Bayes	acc_train_gnb	acc_test_gnb	p_gnb	r_gnb
KNN	acc_train_knn	acc_test_knn	p_knn	r_knn
Decision Tree	acc_train_dt	acc_test_dt	p_dt	r_dt
Random Forest	acc_train_rf	acc_test_rf	p_rf	r_rf
SVM	acc_train_svm	acc_test_svm	p_svm	r_svm
Logistic Reg.	acc_train_lg	acc_test_lg	p_lg	r_lg
ANN	acc_train_ann	acc_test_ann	p_ann	r_ann

Visualizations:
Bar charts comparing accuracy, precision, and recall for all models.

Future Improvements:
Use cross-validation for more robust evaluation.
Implement hyperparameter tuning for better model performance.
Explore advanced deep learning models.
