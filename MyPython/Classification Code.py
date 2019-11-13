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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


#Get the data in!
data = pd.read_csv("path/to/your/data/dat.csv")
print('Data shape', data.shape)

#What are the features? What are their labels? Get rid of rubbish from data
features = data.iloc[:,2:208]
label = data.iloc[:,1]

from sklearn.preprocessing import StandardScaler
df = StandardScaler().fit_transform(features)


# Let's visualise a simple PCA first
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=69)
pca.fit(df)
scores = pca.fit_transform(df)
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
g = sns.scatterplot(x= "PC1", y = "PC2", data=scores_df, hue=label, palette=sns.color_palette("bright", n_colors=3))
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
SVM = SVC(kernel = "rbf", gamma=100, random_state=69)
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=label,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=label, normalize=True,
                      title='Normalized confusion matrix')

plt.show()



#Let's initialise other classifier models before we fit them
LDA = LinearDiscriminantAnalysis()
KN = KNeighborsClassifier(n_neighbors=5)
DecTree = DecisionTreeClassifier()
NBayes = GaussianNB()

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