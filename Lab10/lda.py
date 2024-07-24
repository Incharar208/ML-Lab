# importing the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# loading the dataset
X = load_iris().data
y = load_iris().target

# data preprcessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# calculation and plotting of correlation matrix
correlation_matrix = np.corrcoef(X_scaled.T)
plt.figure(figsize = (6, 6))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title("Correlation Matrix (After Standardization)")
plt.show()

# performing LDA
lda = LinearDiscriminantAnalysis(n_components = 2)
X_projected = lda.fit_transform(X_scaled, y)

# dimensions of original data
print("Shape of data: ", X.shape)
# dimensions of transformed data
print("Shape of transformed data: ", X_projected.shape)

# plotting the results
ld1 = X_projected[:, 0]
ld2 = X_projected[:, 1]
plt.scatter(ld1, ld2, c = y, cmap = "jet")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA of iris dataset")
plt.colorbar()
plt.show()