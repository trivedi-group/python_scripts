from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from statsmodels.multivariate.pca import PCA
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm
matplotlib.style.use('seaborn-poster')

#Get the data!
data = pd.read_csv("path/to/your/data/drift.csv")
print('Data shape', data.shape)

#What are the features? What are their labels? Get rid of rubbish from data
features = data.iloc[:,1:65]
label = data.iloc[:,0]

# Missing value imputation
impute = SimpleImputer()
imputed = impute.fit_transform(features)
# Log Transform dat
logdata = np.log10(imputed)

import statsmodels as stm
from statsmodels.multivariate.pca import PCA
pc = PCA(logdata2, ncomp=10, method='nipals')
scores = pd.DataFrame(pc.scores)
loadings = pd.DataFrame(pc.loadings)

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-poster')
g = sns.scatterplot(x= scores[:0], y = scores[:1], data=scores, palette=sns.color_palette("bright", n_colors=3))
g.set(title="PCA plot",xlabel="PC1", ylabel="PC2")
g.legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.show()




#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(logdata, label, test_size=0.3, random_state=69)
print('Training Features',np.shape(X_train))
print('Training Labels', np.shape(X_test))
print('Testing Features', np.shape(y_train))
print('Testing Labels', np.shape(y_test))

pca=PCA()
pca.plot_scree(logdata)
pca = PCA(data=logdata,ncomp=4, method='nipals')
pca.fit(logdata)
pca.transform(logdata)
scores = pd.DataFrame(pca.scores)
loadings = pd.DataFrame(pca.loadings)
print(scores)
print(loadings)

#plot it away!
g = sns.scatterplot(x= "PC1", y = "PC2", data=pc.scores, hue=label, palette=sns.color_palette("bright", n_colors=2))
g.set(title="PCA plot",xlabel="PC1 "+ "Explained Variance " + xlab +"%", ylabel= "PC2 " + "Explained Variance " + ylab + "%")
g.legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.show()

print('Explained variance:', pc.explained_variance_)
variance = np.round_(pc.explained_variance_, decimals=2)
xlab = np.array2string(pc.explained_variance_[0,])
ylab = np.array2string(pc.explained_variance_[1,])
print(pc.explained_variance_)


#Let's fit the classifiers
SVC = svm.SVC
SVM = SVC(kernel = "rbf", gamma=100, random_state=69)
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))




