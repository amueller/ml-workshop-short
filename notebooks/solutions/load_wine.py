import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

wine = pd.read_csv("https://www.openml.org/data/get_csv/49817/wine_quality.arff")
print(wine.head())

X = wine.drop('quality', axis=1)
y = wine.quality

print("Dataset size: %d  number of features: %d  number of classes: %d"
      % (X.shape[0], X.shape[1], len(np.unique(y))))

X_train, X_test, y_train, y_test = train_test_split(X, y)

pd.plotting.scatter_matrix(X, c=y, cmap='Paired', figsize=(10, 10));
