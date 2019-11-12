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
label = data.iloc[:,1]

print('Dimensions of data', np.shape(data))
dat = data.iloc[:, 1:208]

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
features_scaled = scale.fit(features)
features_scaled_df = pd.DataFrame(features)

#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled_df, label, test_size=0.25, random_state=69)
print('Training Features',np.shape(X_train))
print('Training Labels', np.shape(X_test))
print('Testing Features', np.shape(y_train))
print('Testing Labels', np.shape(y_test))
