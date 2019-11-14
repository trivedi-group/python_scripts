# We will reduce dimensions using PCA and then perform SVM

def pcsvm (features, label, comp):
    "we will perform normal pca and then use results for svm"
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import sklearn
    from sklearn import metrics
    from sklearn import model_selection
    from sklearn import discriminant_analysis
    from sklearn.model_selection import train_test_split
    from statsmodels.multivariate.pca import PCA
    from seaborn import scatterplot
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn import svm
    matplotlib.style.use('seaborn-poster')

    impute = SimpleImputer()
    imputed = impute.fit_transform(features)
    logdata = np.log10(imputed)
    pca = PCA(data=logdata,ncomp=4, method='nipals')
    pca.fit(logdata)
    pca.transform(logdata)
    scores = pd.DataFrame(pca.scores)
    loadings = pd.DataFrame(pca.loadings)
    print(scores)
    print(loadings)

pcsvm(features,label, 4)