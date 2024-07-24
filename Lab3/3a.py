# Visualize the n-dimensional data using heat-map.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Lab3/ToyotaCorolla.csv')

sns.heatmap(data[['Price', 'KM', 'Doors', 'Weight']], cmap = 'jet')

plt.show()