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
