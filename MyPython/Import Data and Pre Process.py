import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

#Get the data!
data = pd.read_csv("path/to/your/data/newerdata.csv")
print('Data shape', data.shape)
data2 = np.transpose(data)
print('Data shape', data2.shape)
data = data2



#What are the features? What are their labels? Get rid of rubbish from data
features = data.iloc[:,2:143]
label = data.iloc[:,0]

df = data3
new_header = df.iloc[0] #grab the first row for the header
df = df[1:] #take the data less the header row
df.columns = new_header #set the header row as the df header

# Missing value imputation
impute = SimpleImputer()
imputed = impute.fit_transform(features)
# Log Transform dat
logdata = np.log10(imputed)

# dummy variables for categorial columns
dummy = pd.get_dummies(label)
dummy = pd.DataFrame(dummy)
logdata = pd.DataFrame(logdata)
df = pd.concat([dummy, logdata], axis=1)
X = logdata
y = label
y = df["pd"]

#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)
print('Training Features',np.shape(X_train))
print('Training Labels', np.shape(X_test))
print('Testing Features', np.shape(y_train))
print('Testing Labels', np.shape(y_test))
