# importing the necessary libararies 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris 

# loading the dataset
iris = load_iris()
data = iris.data[:6]

# calculation euclidean distance between pair of data points
def proximity_matrix(data):
    n = data.shape[0]
    proximity_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            proximity_matrix[i,j] = np.linalg.norm(data[i] - data[j]) 
            proximity_matrix[j,i] = proximity_matrix[i,j]
    return proximity_matrix

# plotting the dendogram
def plot_dendrogram(data, method):
    linkage_matrix = linkage(data, method = method)
    dendrogram(linkage_matrix)
    plt.title(f'Dendrogram - {method} linkage')
    plt.xlabel('Data points')
    plt.ylabel('Distance')
    plt.show()

# plotting correlation matrix
def plot_correlation_matrix(data):
    correlation_matrix = np.corrcoef(data.T)
    plt.figure(figsize = (6,6))
    sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
    plt.title('Correlation Matrix')
    plt.show()

# display the proximity matrix
print("Proximity matrix:")
print(proximity_matrix(data))

# plot dendograms for single and complete linkages
plot_dendrogram(data, 'single')
plot_dendrogram(data, 'complete')

# plot the correlation matrix
plot_correlation_matrix(data)