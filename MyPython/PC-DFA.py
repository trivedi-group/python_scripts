# Load the guns
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from statsmodels.multivariate.pca import PCA
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

#Get the data!
data = pd.read_csv("path/to/your/data/lipids.csv")
print('Data shape', data.shape)
data.head
#What are the features? What are their labels? Get rid of rubbish from data
features = data.iloc[:,2:143]
label = data.iloc[:,1]

# Missing value imputation
impute = SimpleImputer()
imputed = impute.fit_transform(features)
# Log Transform data
logdata = np.log10(imputed)

# dummy variables for categorial columns
dummy = pd.get_dummies(label)
dummy = pd.DataFrame(dummy)
logdata = pd.DataFrame(logdata)
df = pd.concat([dummy, logdata], axis=1)
variables = logdata # This should supercede all 'features' from here on
labels = df["pd"] # This should supercede all 'labels' from here on

# Reduce dimensions, remove collinearity
pc = PCA(logdata, ncomp=2, method='nipals')
scores = pd.DataFrame(pc.scores)
loadings = pd.DataFrame(pc.loadings)

# Prepare data for DFA
X = scores # PCA scores
Y = df["pd"] # this has to be numerical - categorical variable column so change as needed but call it Y
Y= label
# DFA
lda = LDA()
fitted = lda.fit(X,Y)
transformed = lda.fit_transform(X,Y)

# PC meet DFA
matA = pd.DataFrame(fitted.scalings_)
np.shape(matA)
matB = pd.DataFrame(X)
np.shape(matB)

result = np.dot(matB,matA)

plt.plot(transformed[:,0], fitted.scalings_)
plt.show()

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-poster')
g = sns.scatterplot(x= "comp_0", y = "comp_1", data=scores, hue=label, palette=sns.color_palette("bright", n_colors=3))
g.set(title="PCA plot",xlabel="PC1", ylabel="PC2")
g.legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.show()

