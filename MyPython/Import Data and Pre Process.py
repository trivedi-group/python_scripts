import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from numpy import log10
from sklearn.model_selection import train_test_split

#Get the data!
data = pd.read_csv("path/to/your/data/volatiles2.csv")
print('Data shape', data.shape)

#What are the features? What are their labels? Get rid of rubbish from data
features = data.iloc[:,10:681]
label = data.iloc[:,9]

# Missing value imputation
impute = SimpleImputer()
imputed = impute.fit_transform(features)

# Log Transform data
logdata = log10(imputed)

#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(logdata, label, test_size=0.3, random_state=69)
print('Training Features',np.shape(X_train))
print('Training Labels', np.shape(X_test))
print('Testing Features', np.shape(y_train))
print('Testing Labels', np.shape(y_test))
