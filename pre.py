# Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
columns_to_normalize = ['column1', 'column2', 'column3']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Z-Score Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
columns_to_normalize = ['column1', 'column2', 'column3']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

quantitative_data = pd.get_dummies(df.Embarked,prefix = 'Embarked') # one-hot encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

df.dropna() # drop all rows with at least one NaN
df.dropna(axis = 1) # drop all columns with at least one NaN
df.dropna(how = 'all') # drop all rows with all NaN values

df['math_score']=df['math_score'].interpolate()
df['writing_score'] = df['writing_score'].fillna(method='ffill')
df['placement_score'] = df['placement_score'].fillna(method='bfill')
df['club_join_date'] = df['club_join_date'].replace(to_replace=np.nan,value='2019')
df['placement_offer_count']=df['placement_offer_count'].fillna('1')

df['placement_offer_count']=df['placement_offer_count'].astype('int64')
df.pickup_datetime = pd.to_datetime(df.pickup_datetime)

z_scores = np.abs(stats.zscore(data))
threshold = 3 # Define a threshold for considering data points as outliers
outlier_indices = np.where(z_scores > threshold)
data_no_outliers = data[(z_scores <= threshold)]

Q1 = df['math_score'].quantile(0.25)
Q3 = df['math_score'].quantile(0.75)
IQR = Q3 - Q1
lower_lim = Q1 - 1.5*IQR
upper_lim = Q3 + 1.5*IQR
outliers = df[(df['math_score'] < lower_limit) | (df['math_score'] > upper_limit)]
