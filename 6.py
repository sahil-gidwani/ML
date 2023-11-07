# Implement K-Means clustering / hierarchical clustering on sales_data_sample.csv dataset. Determine the number of clusters using the elbow method.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/sahil-gidwani/ML/main/dataset/sales_data_sample.csv", encoding="latin")
df

sns.heatmap(df.corr())

df.dtypes

X = df.iloc[:, [1,4]].values

df.describe()

# mean is far from std this indicates high variance
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)

wcss = []  # Within-Cluster-Sum-of-Squares

# WCSS(K) = Σ d(Ci, Mi)^2
# Where:
# - WCSS(K) is the Within-Cluster-Sum-of-Squares for a specific value of K (number of clusters).
# - Σ represents the sum over all clusters (i = 1 to K), where K is the number of clusters.
# - Ci represents a data point in cluster i.
# - Mi represents the centroid (mean) of cluster i.
# - d(Ci, Mi) is the Euclidean distance between data point Ci and the centroid Mi of its cluster.
# - The goal in K-Means clustering is to minimize the WCSS by finding the optimal number of clusters (K). Typically, the "elbow point" in the plot of WCSS against K is chosen as the optimal number of clusters because it indicates the point where increasing the number of clusters doesn't significantly reduce the WCSS.
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the WCSS values
ks = range(1, 11)
plt.plot(ks, wcss, 'bx-')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()

optimal_k = 3

kmeans_optimal = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42)

# Get the cluster labels for each data point
cluster_labels = kmeans_optimal.fit_predict(X)

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(kmeans_optimal.cluster_centers_[:, 0], kmeans_optimal.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title(f"K-Means Clustering (Number of Clusters: {3})")
plt.legend()
plt.show()

from sklearn.cluster import AgglomerativeClustering

# affinity='euclidean': The 'affinity' parameter determines the distance metric used to measure the dissimilarity between data points. 'euclidean' is a common choice, and it calculates the Euclidean distance between data points in a multi-dimensional space. It's suitable for cases where the data features are continuous and numerical, which is often the case in clustering problems.
# linkage='ward': The 'linkage' parameter specifies the linkage criterion to determine how the distance between clusters is measured. 'ward' is one of the linkage methods. It uses the Ward variance minimization algorithm to measure the distance between clusters, aiming to minimize the increase in variance when two clusters are merged. This method is suitable for hierarchical clustering.
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Create a DataFrame to hold the data and cluster labels
data = pd.DataFrame({'X-axis': X[:, 0], 'Y-axis': X[:, 1], 'Cluster': y_hc})

# Create a scatterplot with different colors for each cluster
sns.scatterplot(data=data, x='X-axis', y='Y-axis', hue='Cluster', palette='viridis')
plt.title('Hierarchical Clustering')
plt.show()

"""
K-Means is a popular clustering algorithm used in machine learning and data analysis to partition a dataset into distinct, non-overlapping groups or clusters based on the similarity of data points. The goal of K-Means clustering is to find natural groupings within the data, where points within the same cluster are more similar to each other than to those in other clusters. It's an unsupervised learning technique, meaning it doesn't require labeled data for training.

Here's how K-Means works:

1. **Initialization**: Choose the number of clusters, K, which represents the number of clusters you want to identify. Additionally, select K initial points as the initial centroids for each cluster. These initial centroids can be randomly chosen from the data or by using specific techniques like "k-means++" for better convergence.

2. **Assignment**: For each data point in the dataset, assign it to the nearest centroid. The "nearness" is typically measured using Euclidean distance, but other distance metrics can be used as well.

3. **Update**: Recalculate the centroids of each cluster as the mean of all data points assigned to that cluster. These centroids represent the new center of their respective clusters.

4. **Reassignment**: Repeat the assignment step using the updated centroids. Each data point is reassigned to the nearest centroid based on the current centroid positions.

5. **Convergence**: Repeat the update and reassignment steps iteratively until a stopping criterion is met. The most common stopping criterion is when the centroids no longer change significantly or when a fixed number of iterations is reached.

6. **Result**: Once the algorithm converges, you have your K clusters, and each data point is assigned to one of these clusters.

K-Means has several advantages and use cases:

- **Simplicity**: It's easy to understand and implement.
- **Efficiency**: It can be efficient on large datasets.
- **Scalability**: It can handle a large number of data points and features.
- **Applicability**: It's suitable for various types of data.

However, K-Means also has limitations:

- **Dependence on Initial Centroids**: The final clusters can be influenced by the initial centroid placement, leading to suboptimal solutions.
- **Sensitive to Outliers**: It is sensitive to outliers, which can significantly affect cluster formation.
- **Dependence on the Number of Clusters**: You need to specify the number of clusters (K) in advance.

To choose the optimal value of K, you can use methods like the Elbow Method, Silhouette Score, or Gap Statistics to find the number of clusters that best represent the data's underlying structure.

In summary, K-Means is a widely used algorithm for clustering data into distinct groups based on similarity, with applications in various fields such as customer segmentation, image compression, and anomaly detection.
"""
"""
Hierarchical clustering is a hierarchical method used in unsupervised machine learning to build a hierarchy of clusters. Unlike K-Means, which requires you to specify the number of clusters (K) in advance, hierarchical clustering doesn't require you to specify K. Instead, it creates a tree-like structure (dendrogram) that represents the relationships between data points at different levels of granularity. It's a versatile clustering method used for various purposes, such as taxonomy, image segmentation, and more.

Here's how hierarchical clustering works:

1. **Initialization**: Treat each data point as an individual cluster. Initially, there are as many clusters as there are data points.

2. **Pairwise Distance Calculation**: Calculate the pairwise distance (similarity or dissimilarity) between all data points. Common distance measures include Euclidean distance, Manhattan distance, or correlation distance, depending on the nature of the data.

3. **Merge Closest Clusters**: Merge the two closest clusters into a new cluster. The choice of which clusters to merge depends on the linkage method, which can be one of the following:
   - Single linkage: Merge clusters based on the minimum pairwise distance between data points from different clusters.
   - Complete linkage: Merge clusters based on the maximum pairwise distance between data points from different clusters.
   - Average linkage: Merge clusters based on the average pairwise distance between data points from different clusters.
   - Ward's method: Merge clusters to minimize the increase in the total within-cluster variance.

4. **Update the Dendrogram**: Update the dendrogram to reflect the merged clusters. The dendrogram illustrates the hierarchy and relationships between clusters.

5. **Repeat**: Repeat the pairwise distance calculation and cluster merging steps iteratively until all data points belong to a single cluster or until a specified stopping criterion is met.

6. **Dendrogram Visualization**: The result is a dendrogram, a tree-like structure that shows the hierarchy of clusters. You can cut the dendrogram at different levels to obtain clusters of varying granularity.

Hierarchical clustering has some advantages:

- **Hierarchy**: It provides a hierarchy of clusters, allowing you to explore data at different levels of detail.
- **No Need for K**: You don't need to specify the number of clusters in advance.
- **Interpretability**: The dendrogram is interpretable and can reveal relationships in your data.

However, it also has limitations:

- **Computationally Intensive**: It can be computationally expensive for large datasets.
- **Sensitivity to Noise**: It's sensitive to noise and outliers.
- **Non-Reversible Merges**: Once clusters are merged, they can't be unmerged.

The choice of linkage method and the decision of where to cut the dendrogram are crucial and can impact the quality of the clustering. You can use criteria like the cophenetic correlation coefficient, silhouette score, or expert knowledge to determine the appropriate level of granularity.

In summary, hierarchical clustering is a flexible clustering method that creates a hierarchy of clusters without requiring the number of clusters as an input. It's useful for exploratory data analysis and revealing underlying structures in data.
"""
