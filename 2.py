"""classify-the-email-using-the-binary-classification.ipynb

Classify the email using the binary classification method. Email Spam detection has two states:
1.   Normal State – Not Spam
2.   Abnormal State – Spam

Use K-Nearest Neighbors and Support Vector Machine for classification. Analyze their performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/sahil-gidwani/ML/main/dataset/emails.csv')
df

df.shape

df.isnull().any()

df.drop(columns = 'Email No.', inplace = True)
df

df.columns

df.Prediction.unique()

df['Prediction'] = df['Prediction'].replace({0 : 'Not spam', 1 : 'Spam'})
df

X = df.drop(columns = 'Prediction', axis = 1)
Y = df['Prediction']

X.columns

Y.head()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# KNN
k_values = []
accuracy_scores = []

best_k = None
best_accuracy = 0.0

# Choose an odd number to avoid tie situations
for k in range(1, 21, 2):  # You can adjust the range based on your preferences
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    k_values.append(k)
    accuracy_scores.append(accuracy)

    # Check if the current k results in a higher accuracy
    if accuracy > best_accuracy:
        best_k = k
        best_accuracy = accuracy

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy Score')
plt.grid(True)
plt.show()

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print(f"Best k: {best_k}")

print("Prediction:")
print(y_pred)

M = metrics.accuracy_score(y_test, y_pred)
print("KNN accuracy: ", M)

C = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=C, display_labels=knn.classes_)
disp.plot()

# SVM Classifier
model = SVC()
# model = SVC(kernel='linear')
# model = SVC(kernel='rbf')  # By default, it uses the 'rbf' kernel
# model = SVC(kernel='poly', degree=3)  # 3rd-degree polynomial kernel
# model = SVC(kernel='sigmoid')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Prediction:")
print(y_pred)

kc = metrics.accuracy_score(y_test, y_pred)
print("SVM accuracy: ", kc)

C = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=C, display_labels=model.classes_)
disp.plot()

# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Prediction:")
print(y_pred)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Logistic Regression accuracy: ", accuracy)

C = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=C, display_labels=model.classes_)
disp.plot()

# [[TN  FP]
#  [FN  TP]]

# # Example confusion matrix
# C = metrics.confusion_matrix(y_test, y_pred)

# # Extract TP, TN, FP, FN
# TN = C[0, 0]  # True Negatives
# FP = C[0, 1]  # False Positives
# FN = C[1, 0]  # False Negatives
# TP = C[1, 1]  # True Positives

# # Calculate accuracy
# accuracy = (TP + TN) / (TP + TN + FP + FN)

# # Calculate precision
# precision = TP / (TP + FP)

# # Calculate recall
# recall = TP / (TP + FN)

# # Calculate specificity (True Negative Rate)
# specificity = TN / (TN + FP)

# # Calculate F1-Score
# f1_score = 2 * (precision * recall) / (precision + recall)

# # Calculate False Positive Rate
# fpr = FP / (FP + TN)

# # Calculate False Negative Rate
# fnr = FN / (FN + TP)

# # Print the calculated metrics
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("Specificity:", specificity)
# print("F1-Score:", f1_score)
# print("False Positive Rate:", fpr)
# print("False Negative Rate:", fnr)

"""
K-Nearest Neighbors (KNN) is a simple and versatile classification and regression machine learning algorithm. It is used for both classification and regression tasks. The key idea behind KNN is to predict the class or value of a data point based on the majority class or average value of its neighboring data points.

Here's how KNN works:

1. **Training Phase:** In the training phase, KNN doesn't learn a specific model. Instead, it memorizes the entire training dataset.

2. **Prediction Phase (Classification):** When you want to classify a new data point, KNN looks at the K-nearest neighbors (data points with the most similar features) from the training dataset. It uses a distance metric (typically Euclidean distance, Manhattan distance, etc.) to measure the similarity between data points.

   - If it's a classification task (predicting a class label), KNN counts the number of neighbors in each class. The class with the majority of neighbors becomes the predicted class for the new data point.

3. **Prediction Phase (Regression):** In regression tasks, KNN predicts a continuous value. Instead of counting class labels, KNN calculates the average (or weighted average) of the target values of the K-nearest neighbors.

   - For example, if you want to predict a house's price, KNN will average the prices of the K-nearest neighboring houses.

4. **Choosing K:** The choice of the value K is crucial in KNN. It's typically an odd number to avoid ties, but the optimal K depends on your dataset and problem. A smaller K (e.g., 1 or 3) makes the model more sensitive to noise, while a larger K provides a smoother decision boundary.

KNN's strengths and weaknesses:

**Strengths:**
- KNN is simple and easy to implement.
- It can be used for both classification and regression.
- It doesn't assume a particular form for the decision boundary.
- It can be effective for multi-class classification problems.

**Weaknesses:**
- KNN can be computationally expensive, especially for large datasets.
- It's sensitive to the choice of the distance metric and the value of K.
- The curse of dimensionality: KNN becomes less effective as the number of dimensions (features) increases.
- It doesn't handle imbalanced datasets well, where one class has significantly more instances than the others.

KNN is often used for its simplicity and can serve as a baseline model for classification and regression tasks. However, to make the most of it, you need to carefully choose the right distance metric, K value, and handle issues like scaling features and handling missing data, if applicable.
"""
"""
Support Vector Machine (SVM) is a supervised machine learning algorithm used for both classification and regression tasks. SVM is particularly popular for classification problems and is known for its ability to find the best separating hyperplane between classes in high-dimensional feature spaces. The primary goal of SVM is to find a hyperplane that maximizes the margin between classes. Here's a detailed explanation of SVM:

**Basic Concepts:**

1. **Hyperplane:** In a two-dimensional space, a hyperplane is a straight line that separates the data into two classes. In a three-dimensional space, it's a flat plane, and in higher dimensions, it's a hyperplane.

2. **Margin:** The margin is the distance between the hyperplane and the nearest data point from either class. SVM aims to maximize this margin.

3. **Support Vectors:** Support vectors are the data points that are closest to the hyperplane and influence the position and orientation of the hyperplane. They are crucial in determining the margin.

4. **Kernel Trick:** SVM can handle non-linearly separable data by using a kernel function that maps the data into a higher-dimensional space where it is linearly separable. Common kernel functions include the linear kernel, polynomial kernel, and radial basis function (RBF) kernel.

**How SVM Works:**

1. **Training Phase:**
   - SVM begins by taking a labeled dataset and aims to find the optimal hyperplane that best separates the data into classes.
   - The goal is to find the hyperplane that maximizes the margin while minimizing the classification error.

2. **Margin Maximization:**
   - The margin is defined as the smallest distance between the hyperplane and any of the support vectors.
   - The SVM optimization problem involves finding the hyperplane weights and bias that maximize this margin.
   - This leads to a constrained optimization problem where SVM attempts to minimize the classification error while keeping the margin as wide as possible.

3. **Kernel Trick:**
   - In cases where data is not linearly separable in the original feature space, SVM can use a kernel function to map the data into a higher-dimensional space.
   - In this higher-dimensional space, a hyperplane can separate the data linearly.

4. **Classification Phase:**
   - Once the optimal hyperplane is found, SVM can be used for classification.
   - Given a new data point, it determines which side of the hyperplane it falls on and assigns it to the corresponding class.

**Strengths and Weaknesses:**

**Strengths:**
- SVM is effective in high-dimensional spaces.
- It works well when the number of features is greater than the number of data points.
- It can handle non-linearly separable data by using appropriate kernels.
- SVM has good generalization properties, making it less prone to overfitting.

**Weaknesses:**
- SVM can be computationally expensive, especially for large datasets.
- It can be sensitive to the choice of the kernel and kernel parameters.
- Interpretability: SVMs are less interpretable compared to decision trees or linear regression.

In summary, SVM is a powerful algorithm for both classification and regression tasks, particularly well-suited to problems where data is high-dimensional. It finds the optimal hyperplane to maximize the margin between classes, and its flexibility is enhanced by the use of kernel functions. However, SVM requires careful tuning of hyperparameters and can be computationally intensive for large datasets.
"""
"""
Logistic Regression is a fundamental machine learning algorithm used for binary classification tasks. It models the relationship between a binary dependent variable (target) and one or more independent variables (features). Despite its name, logistic regression is a classification algorithm, not a regression one.

Here are the key concepts and workings of logistic regression:

1. **Sigmoid Function**: Logistic regression uses a logistic or sigmoid function to model the probability that a given input belongs to a particular class. The sigmoid function maps any real-valued number into a value between 0 and 1. It has an S-shaped curve, which is essential for calculating probabilities.

   The sigmoid function is defined as:
   ```
   S(z) = 1 / (1 + e^(-z))
   ```
   Where `z` is a linear combination of the input features and model parameters:
   ```
   z = b0 + b1 * x1 + b2 * x2 + ... + bn * xn
   ```
   Here, `b0`, `b1`, `b2`, ..., `bn` are the model's coefficients, and `x1`, `x2`, ..., `xn` are the input features.

2. **Hypothesis Function**: The output of the sigmoid function is interpreted as the probability that a given input belongs to the positive class. It is denoted as `P(Y=1|X)` and is expressed as:
   ```
   P(Y=1|X) = 1 / (1 + e^(-z))
   ```
   Here, `Y` is the binary target variable, and `X` represents the feature values.

3. **Decision Boundary**: The model computes the probability that an input belongs to the positive class. To make a binary decision, a threshold (usually 0.5) is set. If the probability is greater than the threshold, the input is classified as the positive class; otherwise, it's classified as the negative class.

4. **Training**: During the training process, the algorithm determines the optimal values of the coefficients (b0, b1, b2, ..., bn) to maximize the likelihood that the observed data is generated by the model. This is typically done using techniques like Maximum Likelihood Estimation or Gradient Descent.

5. **Cost Function**: The performance of the logistic regression model is evaluated using a cost function. In binary classification, a common cost function is the log loss (cross-entropy). The goal during training is to minimize this cost, which measures the difference between the predicted probabilities and the actual class labels.

Advantages of Logistic Regression:
- Simplicity and interpretability: The model's outputs can be directly interpreted as probabilities.
- Efficient and fast for training and prediction.
- Works well for linearly separable data and datasets with a large number of features.

Limitations of Logistic Regression:
- Limited expressiveness: Logistic regression assumes a linear relationship between the features and the log-odds of the target variable. It may not capture complex, nonlinear relationships in the data.
- Sensitive to outliers.
- Not suitable for multi-class classification without modifications (e.g., one-vs-all or softmax regression).

Logistic regression is commonly used for a wide range of binary classification problems, such as spam email detection, disease diagnosis, and customer churn prediction. It serves as a fundamental building block in many machine learning models and remains a valuable tool in data analysis and predictive modeling.
"""
