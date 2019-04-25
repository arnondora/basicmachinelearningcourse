import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Load Iris Sample Dataset
iris = pd.read_csv('../Datasets/Iris.csv')

# Get Column List from Iris
print(iris.columns)

# Pairplot of Iris
sns.pairplot(iris)

# Don't Forget to add this line when we want to plot any graph
plt.show()

# Get Unique Class
print(iris['Species'].unique())

# Show Distribution of each class
sns.countplot(iris['Species'])
plt.show()

# Explore Sepal Length of Each Class
iris_versicolor = iris.loc[lambda data: iris['Species'] == 'Iris-versicolor', :]
iris_setosa = iris.loc[lambda data: iris['Species'] == 'Iris-setosa', :]
iris_virginica = iris.loc[lambda data: iris['Species'] == 'Iris-virginica', :]

sns.distplot(iris_versicolor['SepalLengthCm'], hist=False, color='blue')
sns.distplot(iris_setosa['SepalLengthCm'], hist=False, color='green')
sns.distplot(iris_virginica['SepalLengthCm'], hist=False, color='red')
plt.show()


sns.lmplot(x='SepalLengthCm', y='SepalWidthCm', col='Species', order=2, data=iris)
plt.show()

sns.catplot(x='Species', y='PetalLengthCm', data=iris)
plt.show()

# Relationship Between Sepal Length and Sepal Width
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', size="PetalLengthCm", hue='Species', data=iris)
plt.show()

# Correlation Heatmap
corr = iris.corr()
sns.heatmap(round(corr,2), annot=True)
plt.show()