"""predict-the-price-of-the-uber-ride.ipynb

Predict the price of the Uber ride from a given pickup point to the agreed drop-off location.
Perform following tasks:
1. Pre-process the dataset.
2. Identify outliers.
3. Check the correlation.
4. Implement linear regression and random forest regression models.
5. Evaluate the models and compare their respective scores like R2, RMSE, etc.
"""

!pip install geopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/sahil-gidwani/ML/main/dataset/uber.csv')
df.info()

# 1. Pre-process the dataset
df.shape

df.head()

df.drop(columns=["Unnamed: 0", "key"], inplace=True)
df.head()

df.isnull()

df.isnull().sum()

df['dropoff_latitude'].fillna(value = df['dropoff_latitude'].mean(), inplace = True)
df['dropoff_longitude'].fillna(value = df['dropoff_longitude'].median(), inplace = True)

df.dtypes

# From the above output, we see that the data type of 'pickup_datetime' is 'object
# But 'pickup_datetime'is a date time stamp variable, which is wrongly interpreted as 'object', so we will convert this variable data type to 'datetime'.
df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
df.dtypes

# We will extract time feature from the 'pickup_datetime'
# We will add a variable which measures the distance between pickup and drop
df = df.assign(hour = df.pickup_datetime.dt.hour,
               day = df.pickup_datetime.dt.day,
               month = df.pickup_datetime.dt.month,
               year = df.pickup_datetime.dt.year,
               dayofweek = df.pickup_datetime.dt.dayofweek)

df = df.drop(["pickup_datetime"], axis = 1)
df

from geopy.distance import geodesic

# Filter out rows with valid latitude values
df = df[(df['pickup_latitude'] >= -90) & (df['pickup_latitude'] <= 90) &
              (df['dropoff_latitude'] >= -90) & (df['dropoff_latitude'] <= 90)]

df['dist_travel_km'] = df.apply(
    lambda row: geodesic((row['pickup_latitude'], row['pickup_longitude']),
                        (row['dropoff_latitude'], row['dropoff_longitude'])).kilometers, axis=1)

# 2. Identify outliers
df.plot(kind = "box", subplots = True, layout = (6, 2), figsize = (15, 20))
plt.show()

# Using the Inter Quartile Range to fill the values
def remove_outlier(df1, col):
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5*IQR
    upper_whisker = Q3 + 1.5*IQR
    df[col] = np.clip(df1[col], lower_whisker, upper_whisker)
    return df1

def treat_outliers_all(df1, col_list):
    for c in col_list:
        df1 = remove_outlier(df, c)
    return df1

# df.iloc[:, 0 ::] is a method for selecting rows and columns in a Pandas DataFrame. In this case, : in both positions selects all rows and all columns in the DataFrame df. Essentially, it's selecting the entire DataFrame.
df = treat_outliers_all(df, df.iloc[:, 0 : : ])

df.plot(kind = "box", subplots = True, layout = (7, 2), figsize = (15, 20))
plt.show()

# 3. Check the correlation
# .corr(): This is a Pandas method that is applied to the DataFrame df. When you call df.corr(), it calculates the pairwise correlation coefficients between all pairs of numerical columns in the DataFrame.
# Common correlation coefficients calculated by df.corr() include Pearson's correlation coefficient, Kendall's tau, and Spearman's rank correlation coefficient. By default, df.corr() computes Pearson's correlation coefficient, which is the most widely used.
corr = df.corr()
corr

fig, axis = plt.subplots(figsize = (10, 6))
sns.heatmap(df.corr(), annot = True) # Correlation Heatmap (Light values means highly correlated)

# 4. Implement linear regression and random forest regression models
# Dividing the dataset into feature and target values
df_x = df[['passenger_count', 'hour', 'day', 'month', 'year', 'dayofweek', 'dist_travel_km']]
df_y = df['fare_amount']

# Dividing the dataset into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2)

from sklearn.linear_model import LinearRegression

# Initialize the linear regression model
reg = LinearRegression()

# Train the model with our training data
reg.fit(x_train, y_train)

y_pred_lin = reg.predict(x_test)
print(y_pred_lin)

from sklearn.ensemble import RandomForestRegressor

# Here n_estimators means number of trees you want to build before making the prediction
rf = RandomForestRegressor(n_estimators = 100)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)
print(y_pred_rf)

# 5. Evaluate the models and compare their respective scores like R2, RMSE, etc
cols = ['Model', 'RMSE', 'R-Squared']

# Create a empty dataframe of the colums
# Columns: specifies the columns to be selected
result_tabulation = pd.DataFrame(columns = cols)

from sklearn import metrics
from sklearn.metrics import r2_score

reg_RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lin))
reg_squared = r2_score(y_test, y_pred_lin)

full_metrics = pd.Series({'Model': "Linear Regression", 'RMSE' : reg_RMSE, 'R-Squared' : reg_squared})

# Append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(full_metrics, ignore_index = True)

rf_RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))
rf_squared = r2_score(y_test, y_pred_rf)

full_metrics = pd.Series({'Model': "Random Forest ", 'RMSE': rf_RMSE, 'R-Squared': rf_squared})
# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(full_metrics, ignore_index = True)

# Print the result table
result_tabulation

# # Calculate Mean Error
# mean_error = np.mean(y_test - y_pred_rf)

# # Calculate Mean Absolute Error
# mean_absolute_error = np.mean(np.abs(y_test - y_pred_rf))

# # Calculate Mean Squared Error
# mean_squared_error = np.mean((y_test - y_pred_rf) ** 2)

# # Calculate Root Mean Squared Error
# root_mean_squared_error = np.sqrt(mean_squared_error)

# # Calculate R-squared (R²)
# residuals = y_test - y_pred_rf
# total_sum_of_squares = np.sum((y_test - np.mean(y_test)) ** 2)
# explained_sum_of_squares = np.sum((y_pred_rf - np.mean(y_test)) ** 2)
# r_squared = 1 - (np.sum(residuals ** 2) / total_sum_of_squares)

# # Print the results
# print("Mean Error:", mean_error)
# print("Mean Absolute Error (MAE):", mean_absolute_error)
# print("Mean Squared Error (MSE):", mean_squared_error)
# print("Root Mean Squared Error (RMSE):", root_mean_squared_error)
# print("R-squared (R²):", r_squared)

"""
**Outliers:**

Outliers are data points that significantly differ from the majority of the data in a dataset. They are observations that appear to be inconsistent with the rest of the data and can distort the results of statistical analyses. Outliers can occur for various reasons, including measurement errors, data entry errors, or the presence of rare events. Identifying and handling outliers is crucial in data analysis to ensure that statistical models and analyses are robust and accurate.

**Common Techniques for Handling Outliers:**

1. **Identification and Removal:**
   - Identify outliers using statistical methods, such as the z-score or the IQR (Interquartile Range).
   - Remove or filter out outliers from the dataset. This approach is appropriate when outliers are believed to be the result of errors and should not influence the analysis.

2. **Transformation:**
   - Apply data transformations, such as logarithmic or square root transformations, to reduce the impact of outliers. Transformations can help make the data more normally distributed.

3. **Imputation:**
   - Instead of removing outliers, impute or replace their values with more reasonable estimates. This is commonly used when outliers are believed to be valid data points.

4. **Data Binning:**
   - Divide the data into bins or intervals and categorize the values in each bin. This can help reduce the impact of extreme values, especially in machine learning models.

5. **Clustering:**
   - Use clustering methods to identify and group data points. Outliers may form clusters of their own, making them easier to detect.

6. **Visualizations:**
    - Plot the data to visualize and identify outliers. Tools like box plots and scatter plots are helpful for this purpose.

7. **Expert Knowledge:**
    - Use domain knowledge or consult experts to determine whether an observation is a valid outlier or a meaningful data point.

The choice of outlier handling technique depends on the nature of the data and the goals of the analysis. It's important to carefully consider the implications of each method and the specific context in which the data is being analyzed.
"""
"""
**Correlation:**

Correlation is a statistical measure that quantifies the extent to which two variables are related or associated. In other words, it assesses whether and to what degree changes in one variable coincide with changes in another. Correlation does not imply causation, but it helps to identify relationships between variables in data.

**Pearson's Correlation (Pearson's r):**

Pearson's correlation, often denoted as Pearson's "r," is one of the most widely used methods to measure the strength and direction of a linear relationship between two continuous (numeric) variables. It is a specific type of correlation coefficient. Pearson's r ranges from -1 to 1:

- **Positive correlation (r > 0):** When one variable increases, the other tends to increase as well. It implies a direct linear relationship.

- **Negative correlation (r < 0):** When one variable increases, the other tends to decrease. It implies an inverse linear relationship.

- **No correlation (r = 0):** There's no linear relationship between the variables.

The formula for Pearson's correlation coefficient is:

\[r = \frac{\sum{(X_i - \bar{X})(Y_i - \bar{Y})}}{\sqrt{\sum{(X_i - \bar{X})^2}\sum{(Y_i - \bar{Y})^2}}}\]

Where:
- \(r\) is the Pearson correlation coefficient.
- \(X_i\) and \(Y_i\) are individual data points in the two variables.
- \(\bar{X}\) and \(\bar{Y}\) are the means (average) of the variables X and Y.

**Key Points:**

- Pearson's correlation is specifically designed for linear relationships; it may not capture nonlinear associations.
- It's sensitive to outliers, which can affect the results.
- The range of Pearson's r is from -1 (perfect negative correlation) to 1 (perfect positive correlation).
- A correlation of 0 indicates no linear relationship.
- Positive values indicate that the two variables move together, while negative values indicate they move in opposite directions.
- Pearson's correlation is widely used in fields such as statistics, economics, and social sciences to analyze relationships between variables.

Pearson's correlation is just one type of correlation coefficient. Depending on the data and the nature of the relationship, other correlation measures, such as Spearman's rank correlation or Kendall's tau, might be more appropriate. These measures are used when the relationship between variables is not strictly linear or when you're working with ranked or ordinal data.
"""
"""
1. **Mean Error (ME)**:
   - **Definition**: The mean error measures the average difference between the observed (actual) values and the predicted (estimated) values.
   - **Mathematical Expression**: ME = Σ (actual - predicted) / N, where N is the number of data points.
   - **Interpretation**: A mean error of zero indicates that, on average, the predicted values match the actual values perfectly.

2. **Mean Absolute Error (MAE)**:
   - **Definition**: The mean absolute error calculates the average of the absolute differences between actual and predicted values.
   - **Mathematical Expression**: MAE = Σ |actual - predicted| / N.
   - **Interpretation**: MAE is easy to understand and gives equal weight to all errors.

3. **Mean Squared Error (MSE)**:
   - **Definition**: The mean squared error calculates the average of the squared differences between actual and predicted values.
   - **Mathematical Expression**: MSE = Σ (actual - predicted)^2 / N.
   - **Interpretation**: MSE emphasizes larger errors due to the squaring operation, making it more sensitive to outliers.

4. **Root Mean Squared Error (RMSE)**:
   - **Definition**: The root mean squared error is the square root of the mean squared error.
   - **Mathematical Expression**: RMSE = √(MSE).
   - **Interpretation**: RMSE provides the same unit of measurement as the dependent variable, making it easier to interpret than MSE.

5. **R-squared (R²)**:
   - **Definition**: R-squared, or the coefficient of determination, measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
   - **Mathematical Expression**: R² = 1 - (SSR / SST), where SSR is the sum of squared residuals (Σ (actual - predicted)^2) and SST is the total sum of squares (Σ (actual - mean)^2).
   - **Interpretation**: R² ranges from 0 to 1, with higher values indicating a better fit. An R² of 1 means that all variability is explained by the model, while an R² of 0 indicates that the model doesn't explain any variability.

Each of these evaluation metrics serves a specific purpose:

- **ME** provides the average error, but it doesn't account for the direction (overestimation or underestimation).
- **MAE** gives the average absolute error, making it robust against outliers.
- **MSE** emphasizes larger errors and is widely used in optimization algorithms.
- **RMSE** provides the error in the same units as the dependent variable and is more interpretable than MSE.
- **R²** measures how well the model explains the variability in the data. It is useful for understanding the goodness of fit.

The choice of which metric to use depends on the specific problem, the nature of the data, and the goals of the analysis. Often, a combination of these metrics is used to gain a comprehensive understanding of model performance.
"""
"""
Linear regression is a fundamental statistical and machine learning technique used for predicting a continuous outcome variable (dependent variable) based on one or more independent variables (predictors or features). It establishes a linear relationship between the predictor variables and the target variable. Here's a detailed explanation of linear regression:

**1. Basic Concept:**
   - Linear regression is a supervised learning algorithm for regression tasks.
   - It assumes a linear relationship between the independent variables and the dependent variable.

**2. Simple and Multiple Linear Regression:**
   - In simple linear regression, there's only one predictor variable (X) to predict the target variable (Y).
   - In multiple linear regression, there are multiple predictor variables (X1, X2, X3, ...) to predict the target variable (Y).

**3. Linear Relationship:**
   - The core assumption of linear regression is that the relationship between the predictor variables and the target variable is linear, which can be represented as: `Y = β0 + β1 * X1 + β2 * X2 + ... + βn * Xn`, where Y is the target variable, β0 is the intercept, β1, β2, ..., βn are the coefficients, and X1, X2, ..., Xn are the predictor variables.

**4. Model Parameters:**
   - The coefficients (β0, β1, β2, ...) are the model parameters that need to be estimated during the training process.

**5. Training the Model:**
   - To train a linear regression model, you need a labeled dataset with the target variable and the predictor variables.
   - The model tries to find the best parameters (coefficients) that minimize the sum of squared differences between predicted and actual values. This process is often called "least squares."

**6. Hypothesis Function:**
   - The hypothesis function in linear regression is the linear equation itself: `hθ(x) = β0 + β1 * x1 + β2 * x2 + ... + βn * xn`, where hθ(x) is the predicted value.

**7. Model Evaluation:**
   - After training, you evaluate the model's performance using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared (R2), and others.

**8. Residuals:**
   - Residuals are the differences between actual and predicted values. A well-fitted model has small residuals.

**9. Linearity Assumption:**
   - Linear regression assumes a linear relationship between variables. If this assumption doesn't hold, the model may not perform well.

**10. Multicollinearity:**
   - Multicollinearity occurs when predictor variables are highly correlated. It can affect the model's ability to distinguish the individual effects of variables.

**11. Outliers:**
   - Outliers can heavily influence linear regression models, as they have a significant impact on the fit of the regression line.

**12. Overfitting and Underfitting:**
   - Overfitting occurs when the model is too complex and fits the training data noise. Underfitting occurs when the model is too simple and doesn't capture the underlying patterns.

**13. Regularization:**
   - L1 (Lasso) and L2 (Ridge) regularization techniques can be applied to linear regression to prevent overfitting.

In summary, linear regression is a straightforward yet powerful tool for modeling relationships between variables. It's widely used in various domains, including economics, finance, healthcare, and more, where predicting numerical outcomes is essential.
"""
"""
Random Forest is a versatile and powerful ensemble learning method in machine learning that can be used for both classification and regression tasks. It's particularly popular because of its high predictive accuracy and robustness. Here's a detailed explanation of Random Forest:

**1. Ensemble Learning:**
   - Random Forest is an ensemble learning technique that combines multiple decision trees to make predictions. The idea is that by combining multiple models, you can achieve better predictive performance.

**2. Decision Trees:**
   - Decision trees are the fundamental building blocks of a Random Forest. They're used to split the data into subsets based on the values of the input features. The tree structure resembles a flowchart with nodes and branches.

**3. Randomness in Random Forest:**
   - The "random" in Random Forest comes from two key sources:
      - Random Sampling: For each tree in the forest, a random subset (with replacement) of the training data is used. This process is called "Bootstrap Aggregating" or "Bagging."
      - Random Subset of Features: When constructing each node of the tree, only a random subset of features is considered for splitting. This decorrelates the trees, making them more diverse.

**4. Tree Building:**
   - For each tree in the forest:
      - The data is sampled randomly with replacement (Bootstrapping).
      - A decision tree is constructed on the bootstrapped dataset.
      - At each node of the tree, only a random subset of features is considered for splitting.
      - The process continues until a stopping condition is met (e.g., maximum depth is reached).

**5. Voting or Averaging:**
   - For classification problems, Random Forests use a majority vote among the trees to make predictions. Each tree "votes" for a class, and the class with the most votes is the predicted class.
   - For regression problems, Random Forests average the output (e.g., mean or median) of the individual trees to make a prediction.

**6. Feature Importance:**
   - Random Forests provide a measure of feature importance. It calculates how much each feature contributes to the model's performance by looking at the reduction in impurity (e.g., Gini impurity for classification) achieved when using that feature for splitting nodes.

**7. Robustness:**
   - Random Forests are robust to overfitting, which is a common issue with single decision trees. Combining multiple trees with bootstrapping and random feature selection helps reduce overfitting.

**8. Performance:**
   - Random Forests typically perform well on a wide range of tasks, especially when there are complex relationships between features and the target variable.

**9. Hyperparameters:**
   - Random Forests have hyperparameters that can be tuned, such as the number of trees in the forest, the maximum depth of trees, the size of random feature subsets, and more.

**10. Use Cases:**
   - Random Forests are used in various domains, including classification tasks (e.g., spam detection, medical diagnosis) and regression tasks (e.g., house price prediction).

**11. Python Libraries:**
   - Popular Python libraries, such as scikit-learn, provide easy-to-use implementations of Random Forests.

In summary, Random Forest is a powerful and versatile machine learning technique that leverages the diversity and randomness of decision trees to make accurate and robust predictions. It's a top choice for many data scientists and machine learning practitioners when faced with predictive modeling tasks.
"""
