import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

iris = pd.read_csv('../Datasets/Iris.csv')

# Split Test Train
X = iris.drop('Species', axis=1)
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Naive Bayes
naive_bayes_model = BernoulliNB()
naive_bayes_model.fit(X_train, y_train)

# SVM
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)


# Gradient Boosting
gradient_boosting_model = GradientBoostingClassifier()
gradient_boosting_model.fit(X_train, y_train)

# Show Classification Report

print(classification_report(y_test, naive_bayes_model.predict(X_test)))
print(classification_report(y_test, svm_model.predict(X_test)))
print(classification_report(y_test, gradient_boosting_model.predict(X_test)))

# Show Accuracy Score

print(accuracy_score(y_test, naive_bayes_model.predict(X_test)))
print(accuracy_score(y_test, svm_model.predict(X_test)))
print(accuracy_score(y_test, gradient_boosting_model.predict(X_test)))

algorithm_name = ['Naive Bayes', 'SVM', 'Gradient Boosting']
accuracys = [accuracy_score(y_test, naive_bayes_model.predict(X_test)), accuracy_score(y_test, svm_model.predict(X_test)), accuracy_score(y_test, gradient_boosting_model.predict(X_test))]

sns.set()
sns.barplot(x=algorithm_name, y=accuracys)
plt.show()

# K-Fold Cross Valdidation
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
model = svm.SVC(kernel='linear')
print(cross_validate(model, X, y, cv=k_fold, n_jobs=-1, scoring=['precision_macro', 'recall_macro']))
