#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation

# Context:
# 
# This data set is created only for the learning purpose of the customer segmentation concepts , also known as market basket analysis . I will demonstrate this by using unsupervised ML technique (KMeans Clustering Algorithm) in the simplest form.
# 
# Content:
# 
# You are owing a supermarket mall and through membership cards , you have some basic data about your customers like Customer ID, age, gender, annual income and spending score.
# Spending Score is something you assign to the customer based on your defined parameters like customer behavior and purchasing data.
# 
# Problem Statement:
# 
# You own the mall and want to understand the customers like who can be easily converge [Target Customers] so that the sense can be given to marketing team and plan the strategy accordingly.

# ### Data loading and Exploratory Data Analysis

# Here's what each statistic represents:
# 
# count: The number of non-missing values in each column.
# mean: The average value of the column.
# std: The standard deviation, which measures the spread of values around the mean.
# min: The minimum value in the column.
# 25%: The 25th percentile, also known as the first quartile. It represents the value below which 25% of the data falls.
# 50%: The 50th percentile, also known as the median. It represents the value below which 50% of the data falls.
# 75%: The 75th percentile, also known as the third quartile. It represents the value below which 75% of the data falls.
# max: The maximum value in the column.

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\B.N.KULKARNI\Downloads\archive.zip")

# Display the first few rows of the dataset
print(df.head())

# Check the shape of the dataset
print("Shape of the dataset:", df.shape)

#Data Analysis(shape and type,distribution,correlation of features)

#Check for missing values
print(df.isnull().sum())

#Get statistical summary of the dataset
print(df.describe())

# Data distribution and visualization
numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
categorical_cols = ['Gender']

# Check distribution of numerical columns
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Check distribution of categorical columns (if any)
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
    plt.show()

# Correlation analysis
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# ###  Data Preprocessing(Remove trash,add using feature engineering and transform the format and unclean data)

# Feature scaling is a preprocessing technique used to standardize or normalize the numerical features in a dataset. It ensures that all features have similar scales and ranges, which can be beneficial for many machine learning algorithms.
# 
# The two commonly used methods for feature scaling are:
# 
# 1. **Standardization (Z-score normalization)**: It transforms the features such that they have a mean of 0 and a standard deviation of 1. This method subtracts the mean value from each feature and then divides it by the standard deviation. The formula for standardization is:
# 
#    ```
#    X_scaled = (X - mean(X)) / std(X)
#    ```
# 
#    Standardization retains the shape of the distribution, but it scales and centers the data around zero. It is suitable when the data does not have a Gaussian (normal) distribution.
# 
# 2. **Normalization (Min-Max scaling)**: It scales the features to a fixed range, typically between 0 and 1. This method subtracts the minimum value from each feature and then divides it by the difference between the maximum and minimum values. The formula for normalization is:
# 
#    ```
#    X_scaled = (X - min(X)) / (max(X) - min(X))
#    ```
# 
#    Normalization squeezes the data into a specific range and preserves the relative relationships between values. It is useful when the feature distribution is unknown or when the features have different scales and ranges.
# 
# Feature scaling is important for several reasons:
# 
# 1. **Equalizes feature magnitudes**: Features with significantly different scales can dominate the learning process and bias the model towards those features with larger values. Scaling ensures that all features contribute equally to the analysis.
# 
# 2. **Facilitates gradient-based optimization**: Many machine learning algorithms use gradient-based optimization techniques that require the features to be on a similar scale. Feature scaling helps the optimization process converge more quickly and efficiently.
# 
# 3. **Improves model interpretability**: Scaling the features makes it easier to interpret the model coefficients or feature importances. The standardized or normalized values allow for a more direct comparison of the impact of different features on the model's output.
# 
# It's important to note that feature scaling is not always necessary or beneficial for all algorithms. Some algorithms, such as tree-based models (e.g., decision trees, random forests), are inherently invariant to feature scaling. However, for algorithms that rely on distance-based calculations or gradient descent optimization, feature scaling is typically recommended to achieve optimal performance.

# In[12]:


#Remove irrelevant columns
df = df.drop('CustomerID', axis=1)

#Encode categorical variables (if any) into numerical format using one-hot encoding or label encoding.
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

#Scale the numerical features using feature scaling. In this case, we'll use standardization
from sklearn.preprocessing import StandardScaler

# Extract the relevant features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Perform feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Display the encoded dataset
print(df.head())


# ###  Applying Unsupervised Learning (K-Means Clustering)

# Silhouette Score:
# The silhouette score measures how well each sample in a cluster is separated from samples in other clusters. It provides an indication of the compactness and separation of the clusters. The silhouette score ranges from -1 to 1, where a higher value indicates better-defined and well-separated clusters.
# The formula for calculating the silhouette score of a sample i is as follows:
# 
# css
# Copy code
# silhouette_score(i) = (b(i) - a(i)) / max(a(i), b(i))
# where:
# 
# a(i) is the average distance between i and all other points within the same cluster.
# b(i) is the average distance between i and all points in the nearest neighboring cluster.
# A silhouette score close to 1 indicates that the sample is well-matched to its own cluster and poorly-matched to neighboring clusters. A score close to -1 suggests that the sample is assigned to the wrong cluster, and values around 0 indicate overlapping clusters.
# 
# In the code, the silhouette scores are calculated for different numbers of clusters, and then plotted to identify the number of clusters that yields the highest silhouette score.
# 
# Within-Cluster Sum of Squares (WCSS):
# WCSS represents the sum of squared distances between each sample and its cluster centroid. It measures the compactness of the clusters, with lower values indicating tighter and more homogeneous clusters.
# During the K-means clustering process, the objective is to minimize the WCSS. The algorithm iteratively assigns samples to the nearest centroid and updates the centroids to minimize the sum of squared distances.
# 
# In the code, the WCSS values are calculated for different numbers of clusters using the inertia_ attribute of the KMeans object. The WCSS values are then plotted to identify the "elbow" point, which corresponds to the optimal number of clusters. The elbow point is the point of inflection where adding more clusters does not significantly reduce the WCSS.
# 
# By analyzing the silhouette scores and WCSS, you can determine the appropriate number of clusters for the customer segmentation task

# In[16]:


#Import the necessary libraries
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.mixture import GaussianMixture

# Define a range of cluster numbers to evaluate
min_clusters = 2
max_clusters = 10

# Initialize empty lists for storing evaluation metrics
silhouette_scores = []
wcss = []

# Iterate over different numbers of clusters
for n_clusters in range(min_clusters, max_clusters + 1):
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_features)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(df, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

    # Calculate within-cluster sum of squares (WCSS)
    wcss.append(kmeans.inertia_)

# Plot silhouette scores
plt.figure(figsize=(8, 4))
plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()

# Plot WCSS
plt.figure(figsize=(8, 4))
plt.plot(range(min_clusters, max_clusters + 1), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('WCSS vs Number of Clusters')
plt.show()

#Choose the optimal number of clusters based on the elbow curve and perform k-means clustering
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_features)


# ### Analyzing and Visualizing the Results

# In[21]:


#Assign cluster labels to each customer
df['Cluster'] = kmeans.labels_

#Analyze the resulting clusters
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
df_centers = pd.DataFrame(cluster_centers, columns=features)

#Print the cluster centers
print("Cluster Centers:")
print(df_centers)

#Print the number of customers in each cluster
print("\nNumber of Customers in Each Cluster:")
print(df['Cluster'].value_counts())


# ##### Here are a few ways to enhance the results of the customer segmentation:
# 
# Feature Selection: Analyze and select the most relevant features for customer segmentation. You can use techniques like feature importance or domain knowledge to identify the most informative features.
# 
# Feature Engineering: Create new features that capture additional information about customers. For example, you could calculate the average transaction value, frequency of purchases, or customer loyalty metrics.
# 
# Optimal Number of Clusters: Experiment with different values for the number of clusters and evaluate the results. You can use evaluation metrics like silhouette score or within-cluster sum of squares (WCSS) to choose the optimal number.
# 
# Ensemble Methods: Combine multiple clustering algorithms or perform ensemble clustering to improve the robustness and accuracy of the segmentation.
# 
# Domain-Specific Knowledge: Incorporate domain-specific knowledge to interpret and validate the resulting customer segments. Consider incorporating additional business metrics or qualitative information to further refine the segmentation.
# 
# Remember that customer segmentation is an iterative process, and it may require experimentation and refinement to achieve the best results for your specific business objectives.

# In[ ]:




