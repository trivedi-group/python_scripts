#Let's import everything we need to get going
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import metrics
from sklearn import model_selection
from sklearn import discriminant_analysis
from sklearn.model_selection import train_test_split


# Let's visualise a simple PCA first
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=69)
pca.fit(logdata)
scores = pca.fit_transform(logdata)
scores_df = pd.DataFrame(scores)
scores_df.columns = ["PC1", "PC2"]
scores_df = pd.DataFrame(scores_df)
loadings = pca.components_
loadings_df = pd.DataFrame(loadings) #use these for loadings plot using code below

variance = pca.explained_variance_
print(variance)
variance.shape
variance = np.round_(variance, decimals=2)
xlab = np.array2string(variance[0,])
xlab
ylab = np.array2string(variance[1,])
ylab

from seaborn import scatterplot
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-poster')
g = sns.scatterplot(x= "PC1", y = "PC2", data=scores_df, hue=label, palette=sns.color_palette("bright", n_colors=2))
g.set(title="PCA plot",xlabel="PC1 "+ "Explained Variance " + xlab +"%", ylabel= "PC2 " + "Explained Variance " + ylab + "%")
g.legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.show()


#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.3, random_state=69)
print('Training Features',np.shape(X_train))
print('Training Labels', np.shape(X_test))
print('Testing Features', np.shape(y_train))
print('Testing Labels', np.shape(y_test))

#Let's fit the classifiers
from sklearn.metrics import confusion_matrix
SVM = SVC(kernel = "rbf", gamma=100, random_state=69)
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))



import yellowbrick
from yellowbrick.classifier import ClassificationReport
#visualizer = ClassificationReport(MODEL)
#visualizer.fit_transform(X_train, y_train)
#visualizer.score(X_train,y_train)
#visualizer.show()

from yellowbrick.features import Rank2D
visualizer = Rank2D(algorithm="pearson")
visualizer.fit_transform(features)
visualizer.show()

from yellowbrick.regressor import ResidualsPlot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
visualizer = ResidualsPlot(LinearRegression())
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()


from sklearn.linear_model import Ridge
from yellowbrick.regressor import PredictionError
visualizer = PredictionError(Ridge(alpha=3.181))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()