import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Did you import data and pre process it?
# Have you created your train and test sets?

clf = LDA()
clf.fit(X_train, y_train)
X_train = clf.fit_transform(X_train, y_train)
X_test = clf.transform(X_test)
