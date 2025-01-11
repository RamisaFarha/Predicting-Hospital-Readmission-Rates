# Diabetes Prediction Model

This project aims to predict whether a diabetic patient will be readmitted to a hospital using machine learning. The dataset consists of various medical features and patient history, and the task is to classify whether the patient will be readmitted (binary classification: `1` for readmitted, `0` for not readmitted).

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Modeling](#modeling)
4. [Model Evaluation](#model-evaluation)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Results and Comparison](#results-and-comparison)
7. [Conclusion](#conclusion)

## Project Overview

The goal of this project is to develop a machine learning model that can predict if a diabetic patient will be readmitted to the hospital. We use the dataset from the [Diabetes 130-US hospitals dataset](https://www.kaggle.com/datasets/yangyangz/diabetes-130-us-hospitals) and implement a binary classification task.

We approach this task by preprocessing the data, applying machine learning algorithms, evaluating their performance, and then fine-tuning the best model to maximize accuracy.

## Data Preprocessing

### Steps Taken:
1. **Data Cleaning**: 
    - Handled missing values using the `SimpleImputer` class from `sklearn`. Missing values in categorical features were replaced by the most frequent value, and missing numerical features were replaced by the mean.
   
2. **Feature Encoding**:
    - Applied **One-Hot Encoding** to categorical variables to convert them into a format suitable for machine learning models using `OneHotEncoder` from `sklearn`.
   
3. **Feature Scaling**:
    - Used **Standard Scaling** to normalize the features to a standard scale with a mean of 0 and a standard deviation of 1.

4. **Train-Test Split**:
    - Split the dataset into training and testing subsets using an 80/20 split ratio to evaluate model performance on unseen data.

## Modeling

### Models Used:
1. **Logistic Regression**:
   - A simple yet effective algorithm for binary classification.
   
2. **Random Forest Classifier**:
   - An ensemble learning method that builds multiple decision trees and aggregates their results to improve model performance.
   
3. **Support Vector Machine (SVM)**:
   - A powerful classifier that works well for high-dimensional spaces and is robust to overfitting, especially in higher-dimensional spaces.
   
4. **Gradient Boosting**:
   - An ensemble technique that builds models sequentially, each trying to correct the errors of the previous one.

### Model Training:
We trained the models using the processed training data and evaluated them based on their **accuracy**, **precision**, **recall**, and **f1-score**.

## Model Evaluation

We evaluated all models using the following metrics:
- **Confusion Matrix**: To evaluate the true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Providing precision, recall, and F1-score for each class (readmitted vs. not readmitted).
- **Accuracy Score**: The percentage of correct predictions.

The results showed that the **SVM** model outperformed others with an accuracy of approximately **69%**, while other models like **Logistic Regression** and **Random Forest** showed relatively lower performance.

## Hyperparameter Tuning

We used **GridSearchCV** to tune the hyperparameters of the **SVM** model. After testing different combinations of hyperparameters, we found the best parameters to be:
- **C**: 1
- **Gamma**: 'scale'
- **Kernel**: 'rbf'

This improved the model's performance slightly and made it more generalizable to unseen data.

## Results and Comparison

Here is a summary of the results from each model:

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 62.07%   | 0.64      | 0.60   | 0.62     |
| Random Forest       | 65.52%   | 0.69      | 0.66   | 0.66     |
| SVM                 | 68.97%   | 0.76      | 0.79   | 0.78     |
| Gradient Boosting   | 63.79%   | 0.75      | 0.69   | 0.72     |

- **Best Model**: The **SVM model** with tuned hyperparameters achieved the best performance with an accuracy of **68.97%**, and it performed particularly well in recall for the positive class (readmitted patients).

## Conclusion
The SVM model, after hyperparameter tuning, performed the best among all models for predicting hospital readmission in diabetic patients. Further improvements can be made by exploring additional feature engineering, deploying the model in a real-time environment, or testing it on larger datasets.
