# importing the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

# reading the dataset
df = pd.read_csv('Lab6/glass.csv')

# printing information related to the dataset
print(df.info())
print(df.describe())

# checking for missing values
print("Missing values:\n", df.isnull().sum())

# handling the missing values
imputer = SimpleImputer(strategy = 'median')
df[df.columns] = imputer.fit_transform(df[df.columns])

# normalizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('Type', axis = 1))

# plotting the correlation matrix 
plt.figure(figsize = (10,8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title("Correlation Matrix Heatmap")
plt.show()

# splitting the data
X = X_scaled
y = df['Type'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# defining custom distance functions
def custom_euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def custom_manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# initilising the KNN classfiers
k = 3
clf_custom_euclidean = KNeighborsClassifier(n_neighbors = k, metric = custom_euclidean_distance)
clf_custom_manhattan = KNeighborsClassifier(n_neighbors = k, metric = custom_manhattan_distance)

# training the KNN classifier
clf_custom_euclidean.fit(X_train, y_train)
clf_custom_manhattan.fit(X_train, y_train)

# making the predictions
predictions_euclidean = clf_custom_euclidean.predict(X_test)
predictions_manhattan = clf_custom_manhattan.predict(X_test)

# evaluating the models
accuracy_euclidean = accuracy_score(y_test, predictions_euclidean)
accuracy_manhattan = accuracy_score(y_test, predictions_manhattan)
print("Accuracy with euclidean distance: ", accuracy_euclidean)
print("Accuracy with manhattan distance: ", accuracy_manhattan)

# computing the confusion matrices
cm_euclidean = confusion_matrix(y_test, predictions_euclidean)
cm_manhattan = confusion_matrix(y_test, predictions_manhattan)

# plotting the confusion matrices
plt.figure(figsize = (6, 4))
sns.heatmap(cm_euclidean, annot = True, cmap = 'Blues')
plt.title("Confusion Matrix - Euclidean Distance")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure(figsize = (6, 4))
sns.heatmap(cm_manhattan, annot = True, cmap = 'Blues')
plt.title("Confusion Matrix - Manhattan Distance")
plt.xlabel('Predicted')
plt.ylabel("Actual")
plt.show()