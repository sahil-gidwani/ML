import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
titanic_data = sns.load_dataset('titanic')
df = pd.DataFrame(titanic_data)
df

sns.distplot(df['fare'], bins=20, hist=True, kde=True, rug=True)

sns.jointplot(x = df['age'], y = df['fare'], kind = 'scatter', hue=df['who'])
sns.jointplot(x = df['age'], y = df['fare'], kind = 'hex')

sns.rugplot(x = df['age'], y = df['fare'])

sns.barplot(data = df, x = 'class', y = 'fare')

sns.countplot(data=df, x="class", hue="alive")

sns.histplot(data=df, x="age", kde=True)

sns.boxplot(data=df, x="age", y="class", hue="alive")

sns.violinplot(data=df, x="age", y="class", hue="alive")

sns.stripplot(data=df, x="age", y="class", hue="alive", jitter=False)

sns.swarmplot(data=df, x="age", y="class", hue="alive")

sns.heatmap(df.corr())

sns.clustermap(df.corr())

sns.scatterplot(data=df, x="age", y="fare", hue="class")

sns.lineplot(data=df, x="age", y="fare", hue="class")

sns.pairplot(df)

sns.relplot(data=df, x="age", y="fare", hue="class", kind="scatter")
sns.relplot(data=df, x="age", y="fare", hue="class", kind="line")

survived_counts = titanic_data['survived'].value_counts()
plt.pie(survived_counts, labels=['Not Survived', 'Survived'])
plt.title('Survival Status')
