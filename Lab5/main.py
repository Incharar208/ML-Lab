# importing the necessary packages
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

# loading the dataset
df = pd.read_csv('Lab5/TitanicDataset.csv')
df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# handling the missing values 
imputer = SimpleImputer(strategy='median')
df[['Age', 'Fare']] = imputer.fit_transform(df[['Age', 'Fare']])

# filling missing categorical data 
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)

# encoding categorical data 
label = LabelEncoder()
df['Embarked'] = label.fit_transform(df['Embarked'])

# splitting into X features and y as target variables
X = df.drop('Survived', axis = 1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# initialising and fitting the Gaussian Naive bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# make predictions on the the test set
y_pred = classifier.predict(X_test)

# evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:" , accuracy)