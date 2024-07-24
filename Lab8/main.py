import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

def kmeans(X, K):
    centroids = X[:K]
    pointsPerCentroid = [[] for _ in range(K)]
    for i in range(K, len(X)):
        distances = np.linalg.norm(X[i] - centroids, axis = 1)
        nearestCentroid = np.argmin(distances)
        pointsPerCentroid[nearestCentroid].append(X[i])
        centroids[nearestCentroid] = np.mean(pointsPerCentroid[nearestCentroid], axis = 0)

    labels = np.zeros(X.shape[0])
    print("Labels: ",labels)

    # assigning the samples to the nearest centroids without updating the centroids
    for i in range(len(X)):
        distances = np.linalg.norm(X[i] - centroids, axis = 1)
        nearestCentroid = np.argmin(distances)
        labels[i] = nearestCentroid

    return labels, centroids

# loading the dataset
iris = load_iris()
X = iris.data
y = iris.target

# scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# calculate and plot correlation matrix
correlation_matrix = np.corrcoef(X_scaled.T)
plt.figure(figsize = (6, 6))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title("Correlation Matrix")
plt.show()

K = 3
labels, centroids = kmeans(X_scaled, K)
print("Labels: ", labels)
print("Centroids: ", centroids)

# plotting the clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c = labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', color = 'red', s = 200)
plt.xlabel("Sepal Length(scaled)")
plt.ylabel("Sepal width(scaled)")
plt.title("K-means clustering of Iris Dataset")
plt.show()

# calculation and plotting of confusion matrix
cm = confusion_matrix(labels, y)
plt.figure(figsize = (6, 6))
sns.heatmap(cm, annot = True, cmap = 'Blues', fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# accuracy calculation
accuracy = accuracy_score(labels, y)
print("The accurcay is: ", accuracy)
