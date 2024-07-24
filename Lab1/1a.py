# Visualize the n-dimensional data using 3D surface plots.
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Lab1/ToyotaCorolla.csv')
x = dataset['KM']
y = dataset['Doors']
z = dataset['Price']

ax = plt.axes(projection = '3d')
ax.plot_trisurf(x, y, z, cmap = 'jet')
ax.set_title('3D surface plot')
plt.show()