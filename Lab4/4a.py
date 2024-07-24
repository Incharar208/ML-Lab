# Visualize the n-dimensional data using Box-plot
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Lab4/ToyotaCorolla.csv')

plt.boxplot([data['Price'], data['HP'], data['KM']])
plt.xticks([1,2,3], ['Price', 'Hp', 'KM'])
plt.show()