import pandas as pd
data = pd.read_csv("path/to/your/data/dat.csv")
print('Data shape', data.shape)

features = data.iloc[:,1:207]
label = data.iloc[:,0]

import statsmodels as stm
from statsmodels.multivariate.pca import PCA
pc = PCA(features, ncomp=10, method='nipals')
scores = pd.DataFrame(pc.scores)
loadings = pd.DataFrame(pc.loadings)

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-poster')
g = sns.scatterplot(x= "comp_0", y = "comp_1", data=scores, hue=label, palette=sns.color_palette("bright", n_colors=3))
g.set(title="PCA plot",xlabel="PC1", ylabel="PC2")
g.legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.show()



