# Visualize the n-dimensional data using contour plots
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Lab2/ToyotaCorolla.csv')
x = dataset['KM']
y = dataset['Weight']
z = dataset['Price']

plt.tricontourf(x, y, z, levels = 20, cmap = 'jet')
plt.title('Contour Plot')
plt.xlabel('KM')
plt.ylabel('Weight')
plt.colorbar(label = 'price')
plt.show()