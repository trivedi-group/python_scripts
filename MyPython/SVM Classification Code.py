#Let's import everything we need to get going
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Did you import data and pre process it?
# Have you created your train and test sets?

from sklearn.model_selection import GridSearchCV
from sklearn import svm
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                     'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on training set:" + '%s' % clf.best_params_)

#Let's fit the classifiers
SVM = SVC(kernel = "rbf", gamma=0.001, C=10, max_iter=1, random_state=69)
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))
