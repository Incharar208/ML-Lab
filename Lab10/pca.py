# importing all the necessay packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris 
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

# loading the dataset
X = load_iris().data # contains the feature data
y = load_iris().target # contains the target labels

# data prepocessing 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Fit the scaler to X and transform X to have zero mean and unit variance, storing the result in X_scaled.

# calculation and plotting the correlation matrix
correlation_matrix = np.corrcoef(X_scaled.T)
plt.figure(figsize = (6, 6))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title("Correlation Matrix (After Standardization) ")
plt.show()

# perform PCA using sklearn
pca = SklearnPCA(n_components = 2) # performaing PCA to decompose to 2 principal components
X_projected = pca.fit_transform(X_scaled)

# dimensions of original data
print("Shape of data: ", X.shape)
# dimenisons after PCA transformation
print("Shape of transformed data: ", X_projected.shape)

# plotting the results
pc1 = X_projected[:, 0]
pc2 = X_projected[:, 1]
plt.scatter(pc1, pc2, c = y, cmap = 'jet')
plt.xlabel("Prinicipal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of iris dataset")
plt.colorbar()
plt.show()