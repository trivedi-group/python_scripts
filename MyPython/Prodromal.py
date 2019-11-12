########################### Import required modules ######################################
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('module://backend_interagg')
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree

############################## Import data file - a .csv file##############################
import pandas as pd
data = pd.read_csv("path/to/your/data/dat.csv")
print('Data shape', data.shape)

features = data.iloc[:,1:207]
label = data.iloc[:,0]

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.4, random_state=69)
#print('Training Data Set', X_train.shape)
#print('Test Data Set', X_test.shape)

############ Define what your variables/features are and put it in a separate matrix#######
#variables = data.iloc[:, 10:424]
#variables.replace(0.01, np.nan, inplace=True) #normally change 0.01 to 0
#logdata = np.log10(variables)
#vars = pd.DataFrame(data=logdata[:,:], index = logdata[:,1])
################################# log10 transform data ####################################
#logdata = np.log10(variables)

########### Use multivariate approach to impute values and compare with earlier output#####
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
#imp = IterativeImputer(max_iter=10, random_state=0)
#imputeddata = imp.fit_transform(logdata)
#imputeddataframe = pd.DataFrame(data=imputeddata[:,:])

########### Define what your categorical variable is and put it in a separate matrix/array##
#group = data.loc[:, "phenotype_label"]

#### Put categorial/grouping variable with all necessary variables into one file ####
#Data = pd.concat([imputeddataframe, data[['phenotype_label']]], axis=1)

######################## Standardisation and PCA ##########################################
#pcadata = features
#pcadata.head()
#from sklearn.preprocessing import StandardScaler #only do this if you are not log transforming
#scalar = sklearn.preprocessing.StandardScaler()
#scaleddata = scalar.fit_transform(features)
#np.mean(scaleddata), np.std(scaleddata)

###################### Now calculate principal components ################################
import statsmodels as stm
from statsmodels.multivariate.pca import PCA
pc = PCA(features, ncomp=10, method='nipals')
pc.scores
pc.loadings

from sklearn.decomposition import pca
pca = pca.PCA(n_components=10)
xpca = pca.fit_transform(scaleddata)
scores = xpca
loadings = pca.components_

sns.scatterplot(x= xpca[:,0], y = xpca[:,1], hue=label,  palette=sns.color_palette("colorblind", n_colors=3))
plt.show()

import seaborn as sns
sns.set_palette("colorblind", n_colors=3)
sns.scatterplot(pcaData["principal component 1"],
                pcaData["principal component 2"],
                hue=pcaData["phenotype_label"],
                palette=sns.color_palette("colorblind", n_colors=3))
plt.show()

##################### Data exploration plots with seaborn #####################################
meta = data.iloc[:, 3:8]
meta2 = meta.drop(['sex_label'],axis=1)

sns.pairplot(meta2, hue="phenotype_label", kind="reg")
plt.show()

import seaborn as sns
sns.set(style="white", color_codes=True)
col = sns.set_palette("colorblind")
sns.pairplot(meta2, hue="phenotype_label")
plt.show()
sns.distplot(meta2['Age'])
plt.show()
sns.kdeplot(meta2['Age'], shade=True)
sns.distplot(meta2['Height (m)'])
sns.distplot(meta2['Weight (kg)'])
plt.show()

sns.set_palette("colorblind", n_colors=3)
sns.scatterplot(pcaData["principal component 1"],
                pcaData["principal component 2"],
                hue=pcaData["phenotype_label"],
                palette=sns.color_palette("colorblind", n_colors=3))
plt.show()




