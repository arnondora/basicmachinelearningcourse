import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

# Import Data
iris = pd.read_csv('../Datasets/Iris.csv')

y = iris['Species']
X = iris.loc[:, iris.columns != 'Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

# KMean

predicted_result = KMeans(n_clusters=3, random_state=0).fit(X_train).predict(X_test)
actual_result = y_test.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0,2,1])

# Plot Result

sns.scatterplot(x=actual_result, y=predicted_result, hue=predicted_result)
plt.xlabel('Actual Result')
plt.ylabel('Predicted Result')

plt.show()

print(classification_report(actual_result, predicted_result))