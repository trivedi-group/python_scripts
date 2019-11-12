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

# Get the data in!
data = pd.read_csv("path/to/your/data/dat.csv")
print('Data shape', data.shape)

# What are the features? What are their labels? Get rid of rubbish from data
features = data.iloc[:,2:207]
features = np.log10(features)
label = data.iloc[:,1]

# Creating training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.25, random_state=69)

# Ensemble learning with boostrap validation
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
tree = RandomForestClassifier(n_estimators=10, bootstrap=True, oob_score=True)
tree_model = tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

#important features
imp_features = tree_model.feature_importances_
imp_features_sorted = np.sort(imp_features)
print(np.transpose(imp_features_sorted))




